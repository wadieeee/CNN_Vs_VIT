# CNN vs ViT Benchmark

This project compares **ResNet18 (CNN)** and **DeiT-Tiny (Vision Transformer)** on image classification tasks using PyTorch. It includes automatic experimentation with different learning rates, batch sizes, and optimizers, as well as visualization of feature maps and attention maps.

## Features

- Train and evaluate **ResNet18** and **DeiT-Tiny** from scratch.
- Automatic benchmarking using **YAML configurations**.
- Visualize:
  - CNN feature maps
  - ViT attention maps
- Supports **CPU** or **GPU (CUDA)** training.
- Logs experiments using **TensorBoard**.

## Installation

```bash
pip install -r requirements.txt
