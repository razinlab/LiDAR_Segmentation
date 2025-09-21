# Real-Time 3D LiDAR Segmentation with GPU Acceleration
---
A pipeline that takes raw point cloud data from a LiDAR sensor and can classify each point fast enough to be used in real time. CUDA Kernel to speed up projection algorithm

## Key Features
---
- Custom, fully functional CNN written from scratch
- Ultra-fast spherical projection via a custom CUDA kernel
- Benchmarked on SemanticKITTI dataset
- Achieved a 9.20x speedup in performance compared to NumPy

## Methodology
---
The original plan was to build and speed up a CNN from scratch as a learning project and deep dive into CNNs however, midway through writing the CNN code I realized I didn't have an application in mind to apply the CNN to so I pivoted into a LiDAR segmentation task with the spherical projection step being the one to write a CUDA kernel for. It is more practical to use the PyTorch library as it includes GPU acceleration and write the code for the step that doesn't have native CUDA support.

I constructed a basic encoder-decoder model from scratch and then ported it to PyTorch, and compared the two.

The focus wasn't on accuracy but rather speed.

## Findings
---
CPU training & validation on just 20 samples each took about 98 seconds for 1 epoch. GPU training & validation on about 4000 (give or take) samples each took about 147 seconds for 1 epoch. Extrapolating the epoch time, this points to around a 144x speedup in training speed.

CPU projection on all training sequences (00-07) took around 11.4ms per LiDAR sweep. GPU projection Took about 1.2ms per sweep. 

In terms of frames per second (FPS), we can assume each file to be a LiDAR sweep (frame) and we can calculate the equivalent FPS metric.

CPU: 11.4714 ms → FPS ≈ 1000 / 11.4714 ms ≈ 87.1 FPS

CUDA Accelerated: 1.2473 ms → FPS ≈ 1000 / 1.2473 ≈ 801.7 FPS



