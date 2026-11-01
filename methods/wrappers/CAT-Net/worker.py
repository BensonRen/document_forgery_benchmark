#!/usr/bin/env python3
"""
Persistent worker for CAT-Net v2 inference.

Protocol (stdin/stdout):
  - Startup: prints "READY\n" when model is loaded
  - Per image: reads "<img_path>\t<out_npy_path>\n"
               writes "DONE\n" or "ERROR\n"

Usage (run inside the 'cat' conda env):
    python worker.py <weights_dir> [--device cuda:0]

weights_dir should contain:
    pretrained_models/DCT_djpeg.pth.tar
    pretrained_models/hrnetv2_w48_imagenet_pretrained.pth
    output/splicing_dataset/CAT_full/CAT_full_v2.pth.tar

The benchmark's load_catnet_worker() in run_exp1.py starts this subprocess.
"""
import sys
import os
import argparse
import tempfile
import numpy as np

# ── Resolve repo dir (this file lives at methods/CAT-Net/worker.py) ───────────
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights_dir",
                        help="Directory containing pretrained_models/ and output/ sub-dirs")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    weights_dir = os.path.abspath(args.weights_dir)

    import torch
    import torch.nn.functional as F
    from PIL import Image

    device_str = args.device
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # ── Load CAT-Net config ─────────────────────────────────────────────────────
    # Must chdir to repo so relative paths inside config/yaml work correctly
    saved_cwd = os.getcwd()
    os.chdir(REPO_DIR)

    from lib.config import config as cat_config
    from lib.config import update_config
    from lib import models
    from lib.utils.utils import FullModel
    from lib.core.criterion import CrossEntropy

    yaml_path = os.path.join(REPO_DIR, "experiments", "CAT_full.yaml")
    weights_path = os.path.join(
        weights_dir, "output", "splicing_dataset", "CAT_full", "CAT_full_v2.pth.tar"
    )

    if not os.path.exists(weights_path):
        sys.stderr.write(
            f"[ERROR] CAT_full_v2.pth.tar not found at: {weights_path}\n"
            f"  Expected layout under weights_dir:\n"
            f"    output/splicing_dataset/CAT_full/CAT_full_v2.pth.tar\n"
            f"    pretrained_models/DCT_djpeg.pth.tar\n"
            f"    pretrained_models/hrnetv2_w48_imagenet_pretrained.pth\n"
        )
        sys.stdout.write("ERROR\n")
        sys.stdout.flush()
        sys.exit(1)

    args_ns = type("Args", (), {
        "cfg": yaml_path,
        "opts": [
            "TEST.MODEL_FILE", weights_path,
            "TEST.FLIP_TEST", "False",
            "TEST.NUM_SAMPLES", "0",
        ],
    })()
    update_config(cat_config, args_ns)

    # ── Build model ─────────────────────────────────────────────────────────────
    sys.stderr.write("[INFO] Building CAT-Net model...\n")
    model = eval("models." + cat_config.MODEL.NAME + ".get_seg_model")(cat_config)

    # CrossEntropy criterion (needed only for FullModel wrapper)
    criterion = CrossEntropy(
        ignore_label=cat_config.TRAIN.IGNORE_LABEL,
        weight=torch.FloatTensor([0.5, 0.5]),
    )
    model = FullModel(model, criterion)

    # ── Load checkpoint ─────────────────────────────────────────────────────────
    sys.stderr.write(f"[INFO] Loading weights from {weights_path}\n")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    sys.stderr.write(f"[INFO] Model loaded (epoch {checkpoint.get('epoch', '?')})\n")

    # ── Set up DCT pipeline ─────────────────────────────────────────────────────
    from Splicing.data.AbstractDataset import AbstractDataset

    class _SingleImageDataset(AbstractDataset):
        """Minimal subclass to expose _create_tensor for arbitrary image paths."""

        def __init__(self):
            super().__init__(
                crop_size=None,
                grid_crop=True,
                blocks=("RGB", "DCTvol", "qtable"),
                DCT_channels=1,
            )
            self.tamp_list = []

        def get_tamp(self, index):
            pass  # abstract method — not used here

        def create_tensor_for_path(self, img_path):
            """Return (image_tensor, label_tensor, qtable_tensor) for a single path."""
            im = Image.open(img_path)
            if im.format != "JPEG":
                # CAT-Net requires JPEG for DCT extraction; convert on the fly
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                tmp.close()
                try:
                    im.convert("RGB").save(tmp.name, format="JPEG",
                                           quality=100, subsampling=0)
                    tensors = self._create_tensor(tmp.name, None)
                finally:
                    os.unlink(tmp.name)
            else:
                tensors = self._create_tensor(img_path, None)
            return tensors

    helper = _SingleImageDataset()

    os.chdir(saved_cwd)  # restore working directory

    # ── Signal readiness ────────────────────────────────────────────────────────
    sys.stdout.write("READY\n")
    sys.stdout.flush()

    # ── Main inference loop ─────────────────────────────────────────────────────
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue

        parts = line.split("\t")
        if len(parts) != 2:
            sys.stdout.write("ERROR\n")
            sys.stdout.flush()
            continue

        img_path, out_npy_path = parts[0], parts[1]

        try:
            image, label, qtable = helper.create_tensor_for_path(img_path)
            image = image.unsqueeze(0).to(device)
            label = label.unsqueeze(0).long().to(device)
            if isinstance(qtable, torch.Tensor):
                qtable = qtable.unsqueeze(0).to(device)

            with torch.no_grad():
                _, pred = model(image, label, qtable)

            pred = torch.squeeze(pred, 0)
            pred = F.softmax(pred, dim=0)[1]
            pred_np = pred.cpu().numpy().astype(np.float32)

            np.save(out_npy_path, pred_np)
            sys.stdout.write("DONE\n")
            sys.stdout.flush()

        except Exception as e:
            sys.stderr.write(f"[ERROR] {img_path}: {e}\n")
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.stdout.write("ERROR\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
