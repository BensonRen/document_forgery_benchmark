#!/usr/bin/env python3
"""
Persistent worker for ManTraNet (Wu et al., CVPR 2019).

Protocol (stdin/stdout):
  - Startup: prints "READY\n" once the model is loaded
  - Per image: reads "<img_path>\t<out_npy_path>\n" from stdin,
               writes "DONE\n" or "ERROR\n" to stdout

Usage:
    python worker.py [--weights-dir <path>] [--pretrain-index <0-4>]

Defaults:
    weights-dir: <this script's dir>/pretrained_weights
    pretrain-index: 4  (ManTraNet_Ptrain4.h5, most pretrained)

Keras / TF compatibility shims
-------------------------------
modelCore.py was written for Keras 2.2.0 + TF 1.x.  Running under Keras 2.9
bundled with TF 2.9 requires two categories of fixup:

1. Import-path shims:
   - keras.legacy.interfaces  -> inject a no-op module
   - keras.engine.InputSpec   -> redirect to keras.layers.InputSpec
   - keras.layers.convolutional._Conv -> redirect to base_conv.Conv

2. SRM kernel tracking fix:
   In Keras 2.2, K.variable() inside Layer.build() was NOT automatically
   tracked as a layer weight.  In Keras 2.9 it IS, which causes a weight-count
   mismatch when loading .h5 files (26 saved vs 27 expected).  Fix: patch
   CombinedConv2D._build_SRM_kernel() to return a tf.constant instead of a
   K.variable(), so Keras 2.9 does not count it as a trainable/non-trainable
   weight.
"""

import argparse
import os
import sys
import types

# ── Suppress TF noise ─────────────────────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# ── Locate this script ────────────────────────────────────────────────────────
_WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_WORKER_DIR, "src")


# ── Keras 2.2 → Keras 2.9 compatibility shims ─────────────────────────────────
def _inject_keras_shims():
    import keras
    import keras.engine
    import keras.layers
    import keras.layers.convolutional
    from keras.layers.convolutional.base_conv import Conv as _ConvBase
    from keras.layers import InputSpec as _InputSpec

    # 1. keras.engine.InputSpec
    if not hasattr(keras.engine, "InputSpec"):
        keras.engine.InputSpec = _InputSpec

    # 2. keras.legacy.interfaces  (used by @legacy_conv2d_support decorator)
    if "keras.legacy" not in sys.modules:
        _legacy_mod = types.ModuleType("keras.legacy")
        _interfaces_mod = types.ModuleType("keras.legacy.interfaces")

        def legacy_conv2d_support(func):
            """No-op shim for the @legacy_conv2d_support decorator."""
            return func

        _interfaces_mod.legacy_conv2d_support = legacy_conv2d_support
        sys.modules["keras.legacy"] = _legacy_mod
        sys.modules["keras.legacy.interfaces"] = _interfaces_mod
        keras.legacy = _legacy_mod

    # 3. keras.layers.convolutional._Conv  (private base for conv layers in Keras 2.2)
    _conv_mod = sys.modules.get("keras.layers.convolutional")
    if _conv_mod is not None and not hasattr(_conv_mod, "_Conv"):
        _conv_mod._Conv = _ConvBase


_inject_keras_shims()


# ── SRM kernel patch ───────────────────────────────────────────────────────────
# Must be applied AFTER shims but BEFORE importing modelCore.
def _patch_srm_kernel():
    """Patch CombinedConv2D to store SRM kernel as tf.constant (not K.variable).

    In Keras 2.9, K.variable() created inside Layer.build() is tracked as a
    layer weight, causing a mismatch with .h5 files saved under Keras 2.2 where
    the SRM kernel was not tracked.  tf.constant is not tracked.
    """
    import numpy as np
    import tensorflow as tf

    sys.path.insert(0, _SRC_DIR)
    import modelCore

    def _patched_build_SRM_kernel(self):
        kernel = []
        srm_list = self._get_srm_list()
        for _idx, srm in enumerate(srm_list):
            for ch in range(3):
                ch_kernel = np.zeros([5, 5, 3], dtype="float32")
                ch_kernel[:, :, ch] = srm
                kernel.append(ch_kernel)
        kernel = np.stack(kernel, axis=-1)
        return tf.constant(kernel, dtype=tf.float32, name="srm")

    modelCore.CombinedConv2D._build_SRM_kernel = _patched_build_SRM_kernel


_patch_srm_kernel()

# ── Now import modelCore (already on sys.path from patch step) ────────────────
import modelCore  # noqa: E402
from modelCore import create_model  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="ManTraNet persistent worker")
    p.add_argument(
        "--weights-dir",
        default=os.path.join(_WORKER_DIR, "pretrained_weights"),
        help="Directory containing ManTraNet_Ptrain*.h5 files",
    )
    p.add_argument(
        "--pretrain-index",
        type=int,
        default=4,
        help="Pretrain index (0-4); 4 = most pretrained (default)",
    )
    return p.parse_args()


def load_model(weights_dir, pretrain_index):
    """Build ManTraNet model and load pretrained weights."""
    if pretrain_index == 4:
        imc_model_idx, freeze_featex, window_size_list = 2, False, [7, 15, 31]
    else:
        imc_model_idx, freeze_featex, window_size_list = pretrain_index, False, [7, 15, 31, 63]
    model = create_model(imc_model_idx, freeze_featex, window_size_list)
    weight_file = os.path.join(weights_dir, f"ManTraNet_Ptrain{pretrain_index}.h5")
    if not os.path.isfile(weight_file):
        raise FileNotFoundError(f"Weight file not found: {weight_file}")
    model.load_weights(weight_file)
    return model


_MAX_SIDE = 1024  # cap longest side to avoid TF graph re-tracing on variable-size images


def run_inference(model, img_path):
    """Run ManTraNet on a single image; return float32 2-D score map (H x W)."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    if max(h, w) > _MAX_SIDE:
        scale = _MAX_SIDE / max(h, w)
        img_rgb = cv2.resize(img_rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    inp = np.expand_dims(img_rgb.astype(np.float32) / 255.0, axis=0)  # (1, H, W, 3)
    pred = model.predict(inp, verbose=0)            # (1, H, W, 1)
    score_map = pred.squeeze().astype(np.float32)   # (H, W)
    return score_map


def main():
    args = parse_args()

    # Load model — slow one-time setup
    model = load_model(args.weights_dir, args.pretrain_index)

    # Signal readiness to parent process
    sys.stdout.write("READY\n")
    sys.stdout.flush()

    # Main loop: read "<img_path>\t<out_npy_path>\n", write "DONE\n" or "ERROR\n"
    for raw_line in sys.stdin:
        line = raw_line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            sys.stderr.write(f"[ManTraNet worker] Malformed input: {raw_line!r}\n")
            sys.stdout.write("ERROR\n")
            sys.stdout.flush()
            continue
        img_path, out_npy_path = parts
        try:
            score_map = run_inference(model, img_path)
            np.save(out_npy_path, score_map)
            sys.stdout.write("DONE\n")
        except Exception as exc:
            sys.stderr.write(f"[ManTraNet worker] ERROR on {img_path}: {exc}\n")
            sys.stdout.write("ERROR\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
