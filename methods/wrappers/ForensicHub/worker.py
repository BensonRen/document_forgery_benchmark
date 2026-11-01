#!/usr/bin/env python3
"""
Persistent worker for ForensicHub document models (DTD, FFDN, CAFTB-Net).

Protocol (stdin/stdout):
  - Startup: prints "READY\\n" when model is loaded
  - Per image: reads "<img_path>\\t<out_npy_path>\\n", writes "DONE\\n" or "ERROR\\n"

Usage:
    python worker.py <model_name> <checkpoint_path> [--device cuda:0]
    model_name: dtd | ffdn | caftb
"""

import argparse
import os
import sys
import tempfile
import types

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Mock ForensicHub registry (avoids importing the full package) ──────────────
_fh = types.ModuleType("ForensicHub")
_fh_reg = types.ModuleType("ForensicHub.registry")
_fh_core = types.ModuleType("ForensicHub.core")
_fh_base = types.ModuleType("ForensicHub.core.base_model")


def _register_model(name):
    def d(cls):
        return cls
    return d


class _BaseModel(nn.Module):
    def __init__(self):
        super().__init__()


_fh_reg.register_model = _register_model
_fh_base.BaseModel = _BaseModel
sys.modules["ForensicHub"] = _fh
sys.modules["ForensicHub.registry"] = _fh_reg
sys.modules["ForensicHub.core"] = _fh_core
sys.modules["ForensicHub.core.base_model"] = _fh_base

# ── Add document models to sys.path for relative imports ──────────────────────
_WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_WORKER_DIR, "ForensicHub", "tasks", "document", "models")
sys.path.insert(0, _MODELS_DIR)


# ── JPEG DCT extraction ────────────────────────────────────────────────────────
def _extract_jpeg_dct(img_path, target_hw):
    """Save image as JPEG, read DCT coefficients and QT from it.

    Returns (img_rgb_float32_hwc, dct_int64_hw, qt_int64_8x8).
    """
    import jpegio

    h, w = target_hw
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Use INTER_AREA for downsampling (efficient average pooling; INTER_LANCZOS4 is
    # prohibitively slow on high-res source images like T-SROIE at 4961x7016 px).
    src_h, src_w = img_rgb.shape[:2]
    interp = cv2.INTER_AREA if (src_h > h or src_w > w) else cv2.INTER_LANCZOS4
    img_resized = cv2.resize(img_rgb, (w, h), interpolation=interp)

    # Save as JPEG (quality 75) to extract DCT coefficients.
    # Quality 75 matches the typical document JPEG compression used in the DocTamper
    # training dataset and matches the training augmentation range (75–100).  Using
    # quality 95 produces QT values ~4× smaller than q75, shifting the FPH embedding
    # lookup out-of-distribution and degrading threshold calibration by ~7pp F1.
    fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        cv2.imwrite(
            tmp_path,
            cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 75],
        )
        jpeg = jpegio.read(tmp_path)
        dct_raw = jpeg.coef_arrays[0]  # Y channel, shape (H, W)
        qt_raw = jpeg.quant_tables[jpeg.comp_info[0].quant_tbl_no]  # (8, 8)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Clip absolute DCT values to [0, 20] as in ForensicHub
    dct = np.clip(np.abs(dct_raw), 0, 20).astype(np.int64)  # (H, W)
    qt = qt_raw.astype(np.int64)  # (8, 8)

    img_f = img_resized.astype(np.float32) / 255.0
    return img_f, dct, qt


# ── Image tensor helper ────────────────────────────────────────────────────────
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _to_tensor(img_f_hwc, device):
    """Normalise HWC float32 image to BCHW tensor."""
    img_norm = (img_f_hwc - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)


# ── Model loading ──────────────────────────────────────────────────────────────
def _load_model(model_name, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))

    if model_name == "dtd":
        from dtd.dtd import DTD
        # Monkey-patch backbone init: skip file loading but preserve structural deletions
        # because dtd-9.pth was saved after init_vph/init_swin ran (and deleted sub-modules).
        def _dtd_init_vph(self):
            del self.vph.downsample_layers[3]
            del self.vph.stages[3]
            del self.vph.downsample_layers[2]
            del self.vph.stages[2]

        def _dtd_init_swin(self):
            del self.swin.norm
            del self.swin.patch_embed
            del self.swin.head
            del self.swin.layers[0]

        DTD.init_vph = _dtd_init_vph
        DTD.init_swin = _dtd_init_swin
        model = DTD()
    elif model_name == "ffdn":
        from ffdn.ffdn import FFDN
        # Monkey-patch backbone init: full checkpoint overrides them anyway
        # FFDN.init_vph only loads weights (no structural changes), so checkpoint matches structure
        FFDN.init_vph = lambda self, weight_path=None: None
        model = FFDN()
    elif model_name == "caftb":
        import segmentation_models_pytorch as smp
        from caftb_net.caftb_net import CAFTB_Net

        # Monkey-patch CAFTB_Net.__init__ to build the ViT encoder without a local SMP checkpoint.
        # The full caftb-9.pth checkpoint provides all weights; we just need the right architecture.
        _orig_caftb_init = CAFTB_Net.__init__

        def _patched_caftb_init(self, **kwargs):
            # Temporarily replace smp.from_pretrained so we can build the model structure
            import segmentation_models_pytorch as _smp
            _orig_from_pretrained = _smp.from_pretrained

            class _FakeSegformerModel:
                """Mimics what smp.from_pretrained returns for SegFormer-B5."""
                class _FakeEncoder(nn.Module):
                    def __init__(self):
                        super().__init__()
                        # Build SegFormer-B5 encoder using SMP's get_encoder
                        _enc = _smp.encoders.get_encoder("mit_b5", in_channels=3, depth=4)
                        for name, module in _enc.named_children():
                            setattr(self, name, module)

                encoder = _FakeEncoder()

            def _fake_from_pretrained(path):
                return _FakeSegformerModel()

            _smp.from_pretrained = _fake_from_pretrained
            try:
                _orig_caftb_init(self, **kwargs)
            finally:
                _smp.from_pretrained = _orig_from_pretrained

        CAFTB_Net.__init__ = _patched_caftb_init
        model = CAFTB_Net()
    elif model_name == "tifdm":
        from tifdm.tifdm import Tifdm
        # TIFDM uses timm ResNet50 loaded via timm.create_model(pretrained=True) during __init__.
        # The full tifdm-9.pth checkpoint overrides all weights; suppress pretrained download.
        import timm as _timm
        _orig_create_model = _timm.create_model

        def _no_pretrain_create_model(model_name_arg, **kwargs):
            kwargs["pretrained"] = False
            return _orig_create_model(model_name_arg, **kwargs)

        _timm.create_model = _no_pretrain_create_model
        try:
            model = Tifdm()
        finally:
            _timm.create_model = _orig_create_model
    else:
        raise ValueError(f"Unknown model: {model_name}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WORKER] Missing keys: {len(missing)}", file=sys.stderr)
    model = model.to(device)
    model.eval()
    return model


# ── Inference ─────────────────────────────────────────────────────────────────
def _predict(model, model_name, img_path, device):
    TARGET_HW = (512, 512)

    if model_name in ("dtd", "ffdn"):
        img_f, dct_np, qt_np = _extract_jpeg_dct(img_path, TARGET_HW)
        img_t = _to_tensor(img_f, device)
        dct_t = torch.from_numpy(dct_np).unsqueeze(0).unsqueeze(0).long().to(device)
        qt_t = torch.from_numpy(qt_np).unsqueeze(0).unsqueeze(0).long().to(device)
        mask_t = torch.zeros(1, 1, *TARGET_HW, dtype=torch.long).to(device)
        with torch.no_grad():
            out = model(image=img_t, dct=dct_t, qt=qt_t, mask=mask_t)

    elif model_name in ("caftb", "tifdm"):
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        src_h, src_w = img_rgb.shape[:2]
        interp = cv2.INTER_AREA if (src_h > TARGET_HW[0] or src_w > TARGET_HW[1]) else cv2.INTER_LANCZOS4
        img_resized = cv2.resize(img_rgb, (TARGET_HW[1], TARGET_HW[0]), interpolation=interp)
        img_f = img_resized.astype(np.float32) / 255.0
        img_t = _to_tensor(img_f, device)
        mask_t = torch.zeros(1, 1, *TARGET_HW, dtype=torch.long).to(device)
        with torch.no_grad():
            out = model(image=img_t, mask=mask_t)

    pred = out["pred_mask"]  # (1, 1, H, W) or (1, H, W)
    pred = pred.squeeze().cpu().numpy().astype(np.float32)
    return pred


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["dtd", "ffdn", "caftb", "tifdm"])
    parser.add_argument("checkpoint", help="Path to .pth checkpoint file")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = _load_model(args.model, args.checkpoint, device)

    sys.stdout.write("READY\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        try:
            img_path, out_path = line.split("\t")
            mask = _predict(model, args.model, img_path, device)
            np.save(out_path, mask)
            sys.stdout.write("DONE\n")
        except Exception:
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.stdout.write("ERROR\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
