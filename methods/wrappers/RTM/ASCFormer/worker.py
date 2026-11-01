#!/usr/bin/env python3
"""
Persistent worker for ASCFormer inference.

Reads tab-separated lines from stdin: <img_path>\t<out_npy_path>
Writes "DONE\n" to stdout after each image.

Usage (run inside the 'ascformer' conda env):
    python worker.py <config> <checkpoint> [--device cuda:0]

The benchmark's load_ascformer_model() starts this as a subprocess.
"""
import sys
import os
import argparse
import numpy as np

# ASCFormer uses mmseg custom registry — add repo to path
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="MMSeg config .py file")
    parser.add_argument("checkpoint", help="Model checkpoint .pth file")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch
    import cv2
    from mmengine.config import Config
    from mmseg.apis import init_model
    from mmcv.transforms import Compose

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model = init_model(args.config, args.checkpoint, device=str(device))
    model.eval()

    # Build test pipeline from config, removing LoadAnnotations
    cfg = Config.fromfile(args.config)
    test_pipeline_cfg = [t for t in cfg.test_pipeline
                         if t.get('type') not in ('LoadAnnotations',)]

    pipeline = Compose(test_pipeline_cfg)

    def run_inference(img_path: str) -> np.ndarray:
        """Run ASCFormer on a single image, return float32 prob mask (H, W)."""
        data_ = dict(img_path=img_path)
        data_ = pipeline(data_)

        # Build the batched data dict that data_preprocessor expects
        inputs = [data_['inputs']]
        data_samples = [data_['data_samples']]
        # extras: dict of key -> list of tensors
        extra_dict = data_.get('extra', {})
        extras_batched = {k: [v] for k, v in extra_dict.items()}

        data = dict(inputs=inputs, data_samples=data_samples, extra=extras_batched)

        with torch.no_grad():
            # data_preprocessor handles normalization, stacking, device transfer
            data = model.data_preprocessor(data, training=False)
            results = model.forward(**data, mode='predict')

        # results[0].pred_sem_seg.data: (1, H, W) int64, class 1 = forged
        seg = results[0].pred_sem_seg.data.squeeze(0).cpu().numpy()
        prob = (seg == 1).astype(np.float32)
        return prob

    # Signal readiness
    sys.stdout.write("READY\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            sys.stderr.write(f"[WARN] bad line: {line!r}\n")
            continue

        img_path, out_path = parts
        try:
            prob = run_inference(img_path)
            np.save(out_path, prob)
            sys.stdout.write("DONE\n")
        except Exception as e:
            sys.stderr.write(f"[ERROR] {img_path}: {e}\n")
            sys.stdout.write("ERROR\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
