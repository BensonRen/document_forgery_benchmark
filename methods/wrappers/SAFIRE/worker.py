#!/usr/bin/env python3
"""
Persistent worker for SAFIRE (Segment Any Forged Image Region) inference.

SAFIRE extends SAM (Segment Anything Model) with adaptor layers for image
forgery localization. It performs binary forgery localization by clustering
SAM segment features using k-means (k=2) and identifies the forged cluster.

Reads tab-separated lines from stdin: <img_path>\t<out_npy_path>
Writes "READY\n" to stdout once model is loaded.
Writes "DONE\n" or "ERROR\n" to stdout after processing each image.

Usage (run inside the 'safire' conda env):
    python worker.py --weights /path/to/safire.pth --sam_checkpoint /path/to/sam_vit_b_01ec64.pth [--device cuda:0]

Output:
    Saves a float32 numpy array of shape (H, W) with values in [0, 1]
    representing per-pixel forgery probability.

Reference:
    SAFIRE: Segment Any Forged Image Region, AAAI 2025
    https://github.com/mjkwon2021/SAFIRE
"""
import sys
import os
import argparse
import numpy as np

# Ensure SAFIRE repo root is on path for imports
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True,
                        help="Path to safire.pth checkpoint")
    parser.add_argument("--sam_checkpoint", required=True,
                        help="Path to sam_vit_b_01ec64.pth SAM checkpoint")
    parser.add_argument("--device", default="cuda:0",
                        help="Torch device (default: cuda:0)")
    parser.add_argument("--points_per_side", type=int, default=16,
                        help="Grid density for point prompts (default: 16 -> 16x16 grid)")
    parser.add_argument("--points_per_batch", type=int, default=256,
                        help="Points per inference batch; reduce if OOM (default: 256)")
    args = parser.parse_args()

    import torch
    import cv2
    from PIL import Image

    # ---- Resolve paths ----
    weights_path = os.path.abspath(args.weights)
    sam_ckpt_path = os.path.abspath(args.sam_checkpoint)

    if not os.path.isfile(weights_path):
        sys.stderr.write(f"[SAFIRE] ERROR: weights not found at {weights_path}\n")
        sys.exit(1)
    if not os.path.isfile(sam_ckpt_path):
        sys.stderr.write(f"[SAFIRE] ERROR: SAM checkpoint not found at {sam_ckpt_path}\n")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sys.stderr.write(f"[SAFIRE] Using device: {device}\n")

    # ---- Load model ----
    # Must chdir to repo root so relative imports (segment_anything, networks, etc.) work
    os.chdir(REPO_DIR)

    from segment_anything import sam_model_registry
    from networks.safire_model import AdaptorSAM
    from networks.safire_predictor_binary import SafirePredictor

    sys.stderr.write(f"[SAFIRE] Loading SAM vit_b adaptor from {sam_ckpt_path}\n")
    sam_model = sam_model_registry["vit_b_adaptor"](checkpoint=sam_ckpt_path)
    safire_model = AdaptorSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)

    sys.stderr.write(f"[SAFIRE] Loading SAFIRE weights from {weights_path}\n")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    # Checkpoint is a dict with keys: epoch, model (and possibly optimizer, etc.)
    raw_sd = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    # Strip 'module.' prefix from DDP-trained checkpoints
    new_sd = {k.replace("module.", ""): v for k, v in raw_sd.items()}
    safire_model.load_state_dict(new_sd)
    sys.stderr.write(f"[SAFIRE] Loaded checkpoint (epoch {checkpoint.get('epoch', 'unknown')})\n")

    safire_model.eval()

    # ---- Build the automatic predictor for binary localization ----
    predictor = SafirePredictor(
        safire_model,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=0.0,         # as in infer_binary.py defaults
        stability_score_thresh=0.0,  # keep all segments
        box_nms_thresh=0.0,
    )

    def predict_image(img_path):
        """Run SAFIRE binary forgery localization on one image.

        Returns float32 ndarray of shape (H, W) with per-pixel forgery
        probability in [0, 1], at the original image resolution.
        """
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # SAFIRE/SAM expects uint8 HWC numpy array
        npimage = np.array(img, dtype=np.uint8)

        with torch.no_grad():
            # safire_predict returns (anns, final_pred, max_confidence_indices)
            # final_pred: float32 array of shape (1024, 1024) in [0, 1]
            _anns, final_pred, _indices = predictor.safire_predict(npimage)

        # Resize prediction back to original image dimensions
        if final_pred.shape != (orig_h, orig_w):
            final_pred = cv2.resize(
                final_pred.astype(np.float32),
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR,
            )

        return final_pred.astype(np.float32)

    # ---- Signal ready ----
    sys.stdout.write("READY\n")
    sys.stdout.flush()
    sys.stderr.write("[SAFIRE] Worker ready. Waiting for image paths on stdin.\n")
    sys.stderr.flush()

    # ---- Main inference loop ----
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        parts = line.split("\t")
        if len(parts) != 2:
            sys.stderr.write(f"[SAFIRE] Bad input line (expected 2 tab-separated fields): {line!r}\n")
            sys.stderr.flush()
            continue

        img_path, out_path = parts
        img_path = os.path.abspath(img_path)
        try:
            prob = predict_image(img_path)
            np.save(out_path, prob)
            sys.stdout.write("DONE\n")
        except Exception as e:
            sys.stderr.write(f"[SAFIRE] ERROR processing {img_path}: {e}\n")
            import traceback
            sys.stderr.write(traceback.format_exc())
            sys.stdout.write("ERROR\n")
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()
