#!/usr/bin/env python3
"""
Persistent worker for ADCD-Net inference.

Reads tab-separated lines from stdin: <img_path>\t<out_npy_path>
Writes "DONE\n" to stdout after each image.

Usage (run inside the 'adcdnet' conda env):
    python worker.py <weights_dir> [--device cuda:0]

weights_dir should contain: ADCDNet.pth, docres.pkl, DocTamperOCR.zip (already extracted)

The benchmark's load_adcdnet_model() starts this as a subprocess.
"""
import sys
import os
import argparse
import tempfile
import numpy as np

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights_dir", help="Directory with ADCDNet.pth and docres.pkl")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    import cv2
    from PIL import Image
    import pickle

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # Resolve weights_dir to absolute path BEFORE any os.chdir()
    weights_dir = os.path.abspath(args.weights_dir)

    # ---- Load model ----
    # Set working dir to repo so cfg.py imports work
    os.chdir(REPO_DIR)
    import cfg as cfg_module
    # Patch cfg paths to actual weight locations before model instantiation
    cfg_module.docres_ckpt_path = os.path.join(weights_dir, "docres.pkl")
    cfg_module.ckpt = os.path.join(weights_dir, "ADCDNet.pth")
    from model.model import ADCDNet

    model = ADCDNet()
    ckpt_path = os.path.join(weights_dir, "ADCDNet.pth")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Bug fix (2026-02): state is a training checkpoint dict with keys
    # ('model', 'optimizer', 'step', 'scheduler').  The original code
    # iterated state.items() directly, which yielded checkpoint-level keys
    # (not weight tensors), so load_state_dict silently left all parameters
    # at random initialisation — causing pixel_auc ≈ 0.499.
    # Fix: extract the nested model state dict first, then strip module. prefix.
    raw_sd = state["model"] if isinstance(state, dict) and "model" in state else state
    new_sd = {}
    for k, v in raw_sd.items():
        new_sd[k.replace("module.", "")] = v
    model.load_state_dict(new_sd, strict=False)
    model = model.to(device)
    model.eval()

    # ---- Set up OCR (PaddleOCR) ----
    # Note: docres is loaded by ADCDNet() constructor via cfg.docres_ckpt_path
    try:
        from seg_char import get_ocr_mask
        has_ocr = True
    except Exception as e:
        sys.stderr.write(f"[WARN] PaddleOCR unavailable: {e}; using empty OCR masks\n")
        has_ocr = False

    # ---- Set up JPEG DCT ----
    try:
        from jpeg2dct.numpy import load as jpeg2dct_load
        has_dct = True
    except ImportError:
        sys.stderr.write("[WARN] jpeg2dct unavailable; using zero DCT\n")
        has_dct = False

    import torchvision.transforms as T
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    INFER_SIZE = 512

    def load_image_tensors(img_path):
        """Returns (img_tensor, dct_tensor, qt_tensor, ocr_mask_tensor) for one image."""
        img = Image.open(img_path).convert("RGB")
        ori_w, ori_h = img.size
        img_r = img.resize((INFER_SIZE, INFER_SIZE), Image.LANCZOS)

        # RGB tensor
        img_t = T.ToTensor()(img_r)
        img_t = normalize(img_t).unsqueeze(0).to(device)  # (1,3,H,W)

        # DCT tensor: ADCD-Net expects (1, H, W) Long tensor with values in [0, 20]
        # computed as: abs(pixel-wise DCT) clipped to [0, 20]
        # QT tensor: (1, 8, 8) Long tensor of JPEG luma quantization table values
        if has_dct:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                img_r.save(tmp_path, "JPEG", quality=75)
            try:
                # Load Y-channel DCT from JPEG: shape (H/8, W/8, 64)
                dct_y, _, _ = jpeg2dct_load(tmp_path, normalized=False)
                rows, cols, _ = dct_y.shape
                # Reconstruct full-image pixel-wise DCT (H, W)
                dct_full = np.empty((8 * rows, 8 * cols), dtype=np.int32)
                for j in range(rows):
                    for i in range(cols):
                        dct_full[8*j:8*(j+1), 8*i:8*(i+1)] = dct_y[j, i].reshape(8, 8)
                # abs and clip to [0, 20] matching training preprocessing
                dct_full = np.clip(np.abs(dct_full), 0, 20).astype(np.int64)
                dct_t = torch.from_numpy(dct_full).unsqueeze(0).to(device)  # (1, H, W)

                # Luma QT from JPEG: (8, 8) integer values clipped to [0, 63]
                from get_qt import get_luma_qt_8x8
                qt_arr = get_luma_qt_8x8(tmp_path)
                if qt_arr is None:
                    qt_arr = np.ones((8, 8), dtype=np.int64)
                qt_arr = np.clip(qt_arr, 0, 63).astype(np.int64)
                qt_t = torch.from_numpy(qt_arr).unsqueeze(0).to(device)  # (1, 8, 8)
            finally:
                os.unlink(tmp_path)
        else:
            dct_t = torch.zeros(1, INFER_SIZE, INFER_SIZE, dtype=torch.long, device=device)
            qt_t = torch.zeros(1, 8, 8, dtype=torch.long, device=device)

        # OCR mask: (1, 1, H, W) binary, 1 = text region
        if has_ocr:
            try:
                ocr_mask = get_ocr_mask(img_r)  # returns H×W numpy array, 0/1
                ocr_t = torch.from_numpy(ocr_mask).float().unsqueeze(0).unsqueeze(0).to(device)
            except Exception:
                ocr_t = torch.zeros(1, 1, INFER_SIZE, INFER_SIZE, device=device)
        else:
            ocr_t = torch.zeros(1, 1, INFER_SIZE, INFER_SIZE, device=device)

        return img_t, dct_t, qt_t, ocr_t, ori_h, ori_w

    # Signal readiness
    sys.stdout.write("READY\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue

        img_path, out_path = parts
        try:
            img_t, dct_t, qt_t, ocr_t, ori_h, ori_w = load_image_tensors(img_path)
            with torch.no_grad():
                logits = model(img_t, dct_t, qt_t, None, ocr_t, is_train=False)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                # logits: (1, 2, H, W) — class-1 is forged
                prob = F.softmax(logits, dim=1)[:, 1, :, :].squeeze().cpu().numpy()
            if prob.shape != (ori_h, ori_w):
                prob = cv2.resize(prob, (ori_w, ori_h))
            np.save(out_path, prob.astype(np.float32))
            sys.stdout.write("DONE\n")
        except Exception as e:
            sys.stderr.write(f"[ERROR] {img_path}: {e}\n")
            sys.stdout.write("ERROR\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
