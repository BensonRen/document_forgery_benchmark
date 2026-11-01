# ManTraNet Setup Notes

## Overview

ManTraNet (Wu et al., CVPR 2019) is an end-to-end image forgery detection and
localization network.  It was written for Keras 2.2.0 + TensorFlow 1.x, which
is incompatible with the main DocForge-Bench venv (Keras 3.x).

We isolate ManTraNet in a dedicated conda environment and run it via a
persistent worker subprocess, following the same pattern used for ASCFormer,
ADCD-Net, and other incompatible methods.

## Conda Environment

Environment name: **mantranet**
Location: `~/.conda/envs/mantranet/`

### Creation commands

```bash
# Create Python 3.8 environment
conda create -y -n mantranet python=3.8

# Install TF 2.9.3 (bundles Keras 2.9, compatible with modelCore.py after shims)
~/.conda/envs/mantranet/bin/pip install tensorflow==2.9.3

# Install OpenCV (headless, no GUI deps)
~/.conda/envs/mantranet/bin/pip install opencv-python-headless
```

### Key package versions

| Package | Version |
|---------|---------|
| Python | 3.8 |
| tensorflow | 2.9.3 |
| keras | 2.9.0 (bundled with TF) |
| numpy | 1.24.4 (TF dependency) |
| opencv-python-headless | 4.13.x |

## Compatibility Shims

`modelCore.py` (Keras 2.2 API) requires three import-path shims and one
structural patch to work under Keras 2.9:

### 1. `keras.legacy.interfaces` (missing module)

Keras 2.9 removed `keras.legacy`. We inject a no-op module:

```python
import types, sys
_legacy_mod = types.ModuleType("keras.legacy")
_interfaces_mod = types.ModuleType("keras.legacy.interfaces")
_interfaces_mod.legacy_conv2d_support = lambda f: f  # no-op decorator
sys.modules["keras.legacy"] = _legacy_mod
sys.modules["keras.legacy.interfaces"] = _interfaces_mod
keras.legacy = _legacy_mod
```

### 2. `keras.engine.InputSpec` (moved)

In Keras 2.9, `InputSpec` lives in `keras.layers`, not `keras.engine`:

```python
from keras.layers import InputSpec
import keras.engine as _ke
_ke.InputSpec = InputSpec
```

### 3. `keras.layers.convolutional._Conv` (private class removed)

In Keras 2.2, `_Conv` was the private base for all conv layers.
In Keras 2.9, the equivalent is `keras.layers.convolutional.base_conv.Conv`:

```python
from keras.layers.convolutional.base_conv import Conv as _ConvBase
import keras.layers.convolutional as _conv_mod
_conv_mod._Conv = _ConvBase
```

### 4. SRM kernel weight tracking (structural mismatch)

In Keras 2.2, calling `K.variable()` inside `Layer.build()` did NOT
automatically add the result to the layer's tracked weights.  In Keras 2.9 it
DOES.  `CombinedConv2D._build_SRM_kernel()` creates the SRM filter bank with
`K.variable()`, causing Keras 2.9 to track it as a layer weight — resulting in
a weight-count mismatch (26 weights in `.h5` vs 27 expected by Keras 2.9).

Fix: monkey-patch `_build_SRM_kernel` to return a `tf.constant` instead:

```python
import tensorflow as tf
import numpy as np

def _patched_build_SRM_kernel(self):
    kernel = []
    for srm in self._get_srm_list():
        for ch in range(3):
            k = np.zeros([5, 5, 3], dtype='float32')
            k[:, :, ch] = srm
            kernel.append(k)
    kernel = np.stack(kernel, axis=-1)
    return tf.constant(kernel, dtype=tf.float32, name='srm')  # not tracked

modelCore.CombinedConv2D._build_SRM_kernel = _patched_build_SRM_kernel
```

## Weights

Pretrained weights are in `pretrained_weights/`:

| File | Pretrain type | Notes |
|------|--------------|-------|
| `ManTraNet_Ptrain0.h5` | IMC only | Baseline |
| `ManTraNet_Ptrain1.h5` | + copy-move | |
| `ManTraNet_Ptrain2.h5` | + splicing | |
| `ManTraNet_Ptrain3.h5` | + removal | |
| `ManTraNet_Ptrain4.h5` | + enhancement | Most pretrained — **default** |

## Worker Protocol

`worker.py` implements the standard DocForge-Bench worker protocol:

- **Startup**: loads model, prints `READY\n` to stdout
- **Per image**: reads `<img_path>\t<out_npy_path>\n` from stdin,
  runs inference, saves a float32 `.npy` score map (pixel values 0–1,
  higher = more likely manipulated), writes `DONE\n` or `ERROR\n` to stdout

## Integration

- **Loader**: `load_mantranet()` in `scripts/run_exp1.py` —
  calls `_make_subprocess_worker_loader()` with `env_name="mantranet"`.
- **Config**: `configs/exp1_general_on_docs.yaml` — ManTraNet
  is active with `weights: "pretrained_weights/ManTraNet_Ptrain4.h5"` and
  `pretrain_index: 4`.

## Testing

Quick smoke test (requires the mantranet conda env and weights):

```bash
python3 -c "
import subprocess, os, numpy as np
worker = 'methods/ManTraNet/worker.py'
python = '~/.conda/envs/mantranet/bin/python'
proc = subprocess.Popen([python, worker], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
# wait for READY
while True:
    line = proc.stdout.readline().decode().strip()
    if line == 'READY': break
img = 'methods/ManTraNet/data/forged/I00_dgzb67i_0.jpg'
out = '/tmp/test_mantranet.npy'
proc.stdin.write(f'{img}\t{out}\n'.encode()); proc.stdin.flush()
status = proc.stdout.readline().decode().strip()
print('Status:', status)
arr = np.load(out); print('Shape:', arr.shape, 'Range:', arr.min(), arr.max())
proc.stdin.close(); proc.wait()
"
```

Expected output: `Status: DONE` followed by shape and a range within [0, 1].
