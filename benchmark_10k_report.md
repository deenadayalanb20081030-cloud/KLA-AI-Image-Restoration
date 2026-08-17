# 10,000-Wafer Tile Multi-Model Metrology Benchmark Report

**Dataset Scale**: 10,000 Paired Semiconductor Images across 8 Fab Categories (.png + .npy)
**Hardware Platform**: NVIDIA H100 Tensor Core GPU (80GB SXM5)
**Total Benchmark Execution Time**: 139.23 s

## Quantitative Performance Comparison Table

| Model Architecture | Mean PSNR (dB) | Mean SSIM | ΔLER Line-Edge Roughness | H100 Latency | Throughput (FPS) | Fab Capacity (Tiles/Min) | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NAFNet-Metrology (Ours)** | 20.99 dB | 0.8322 | 0.29 nm | 3.8 ms | 263.2 FPS | ~15,789 Tiles/min | 2.2 GB |
| Restormer-Lite | 20.87 dB | 0.8278 | 0.28 nm | 7.4 ms | 135.1 FPS | ~8,108 Tiles/min | 4.6 GB |
| SwinIR-Metrology | 20.8 dB | 0.8252 | 0.28 nm | 9.1 ms | 109.9 FPS | ~6,593 Tiles/min | 5.8 GB |
| UNet-Baseline | 20.4 dB | 0.8141 | 0.28 nm | 4.5 ms | 222.2 FPS | ~13,333 Tiles/min | 2.8 GB |


## Visual Comparison Outputs

Visual paired comparison sets for all 4 models saved to: `outputs/comparison_10k/`
