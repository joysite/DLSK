[English](../README.md) | [简体中文](README.zh-CN.md) 

<div align="center">

# DLSK

**Directional Large Selective Kernel**

A PyTorch module for directional feature modeling in computer vision.

</div>

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

DLSK combines local depthwise convolution, vertical and horizontal strip
convolutions, and spatial attention that predicts three branch weights. It is
intended as an experimental feature-refinement module for vision tasks where
directional or elongated patterns may matter.

Potential application areas include crop-organ detection, crack inspection,
road extraction, vessel segmentation, and text-line analysis. These are
candidate use cases, not a claim that performance has been validated on every
task.

## Module design

For an input feature map `x` with shape `(B, C, H, W)`:

1. A `3×3` depthwise convolution extracts local features.
2. A dilated `11×1` depthwise convolution models vertical context.
3. A dilated `1×11` depthwise convolution models horizontal context.
4. The three features are summed and passed through a `1×1` fusion convolution.
5. Channel-wise average and maximum descriptors are concatenated and passed to
   a `5×5` convolution to predict three spatial attention maps.
6. The three branch features are weighted and summed.

**Implementation note:** the vertical and horizontal convolutions are applied
sequentially to the local feature map in the current implementation. They are
not independent parallel branches. The attention convolution is a standard
`Conv2d(2, 3, 5)`, not a depthwise convolution.

## Installation

```bash
git clone https://github.com/joysite/DLSK.git
cd DLSK
python -m pip install -r requirements.txt
```

Install the project in editable mode for development:

```bash
python -m pip install -e ".[dev]"
```

## Quick start

```python
import torch
from dlsk import DLSK, DLSKBlock

x = torch.randn(1, 64, 80, 80)

module = DLSK(dim=64)
y = module(x)
print(y.shape)  # torch.Size([1, 64, 80, 80])

block = DLSKBlock(dim=64)
z = block(x)
print(z.shape)  # torch.Size([1, 64, 80, 80]
```

## Integration notes

DLSK preserves the input channel count and spatial dimensions, so it can be
inserted at suitable points in a CNN when the surrounding model expects the
same tensor shape. Actual integration into YOLO or another framework may
require framework-specific module registration, parser updates, and model
configuration changes. No framework integration is included in this minimal
repository.

## Tests

Run the unit tests with:

```bash
pytest -q
```

The tests check output shapes, gradient propagation, invalid inputs, and the
`DLSKBlock` wrapper. They do not establish detection accuracy, speed, or
superiority over other modules.

## Reporting results

When publishing benchmark results, specify the dataset and split, baseline,
training schedule, random seed, hardware, input size, parameter count, FLOPs
measurement method, and inference settings. Avoid claiming that DLSK improves
accuracy or speed until those results have been measured and reproducibly
verified.

## Current limitations

- The attention maps use independent sigmoid gates; their values are not
  constrained to sum to one.
- The directional convolutions are sequential in this implementation.
- Performance and latency depend on the host architecture and deployment
  environment.
- This repository does not include pretrained weights or a complete detector.

## Citation

If you use DLSK in academic work, cite the repository version you used and
provide the relevant paper citation if a formal publication becomes available.
Do not use placeholder citation metadata as though it were a published paper.

## License

Distributed under the MIT License. See [LICENSE](LICENSE).
