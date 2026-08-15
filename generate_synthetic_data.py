"""
KLA AI Hackathon: Helper Utility to Generate Synthetic Test Datasets
Creates realistic degraded semiconductor test samples in ./sample_test_data/input/
"""

import os
import numpy as np
from PIL import Image


def generate_sample_wafer_texture(width=256, height=256):
    """Generates Figure 1 Texture pattern (512x512 downsampled to 256x256 with speckle)"""
    y, x = np.mgrid[0:height, 0:width]
    u = x / width
    v = y / height
    
    n = np.sin(u * 80 + np.sin(v * 40)) * 0.15
    n += np.cos(v * 70 + np.cos(u * 50)) * 0.15
    n += np.sin((u + v) * 120) * 0.1
    
    base = 0.4 + n * 0.5
    # Apply multiplicative speckle (>1.0 intensity spread)
    speckle = np.random.gamma(shape=10.0, scale=0.1, size=(height, width))
    noisy = base * speckle + np.random.normal(0, 0.02, size=(height, width))
    
    # Save as standard 8-bit image with high dynamic range values
    img_uint8 = np.clip(noisy * 255.0, 0, 255).astype(np.uint8)
    return img_uint8


def generate_sample_dendrite(width=128, height=128):
    """Generates Figure 2 Dendrite crystal defect pattern (256x256 downsampled to 128x128 with speckle)"""
    arr = np.full((height, width), 10, dtype=np.float32)
    cx, cy = width // 2, height // 2
    
    for i in range(12):
        angle = (i / 12) * np.pi * 2
        for r in range(15, 55):
            px = int(cx + r * np.cos(angle) + np.sin(r * 0.2) * 3)
            py = int(cy + r * np.sin(angle) + np.cos(r * 0.2) * 3)
            if 0 <= px < width and 0 <= py < height:
                arr[py, px] = 220
                if r % 4 == 0 and px + 1 < width and py + 1 < height:
                    arr[py+1, px+1] = 180
                    
    speckle = np.random.gamma(shape=8.0, scale=0.125, size=(height, width))
    noisy = (arr / 255.0) * speckle
    img_uint8 = np.clip(noisy * 255.0, 0, 255).astype(np.uint8)
    return img_uint8


def main():
    out_dir = "./sample_test_data/input"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"[INFO] Generating synthetic test images in {out_dir}...")
    
    # 1. Generate 256x256 texture samples (to be restored to 512x512)
    for i in range(4):
        img_arr = generate_sample_wafer_texture(256, 256)
        fname = f"sample_texture_{i+1:03d}_256x256.png"
        Image.fromarray(img_arr).save(os.path.join(out_dir, fname))
        print(f"  -> Created {fname} (256x256)")
        
    # 2. Generate 128x128 dendrite samples (to be restored to 256x256)
    for i in range(4):
        img_arr = generate_sample_dendrite(128, 128)
        fname = f"sample_dendrite_{i+1:03d}_128x128.png"
        Image.fromarray(img_arr).save(os.path.join(out_dir, fname))
        print(f"  -> Created {fname} (128x128)")
        
    print(f"[SUCCESS] Created 8 test images in {out_dir}")
    print(f"Test evaluation with:")
    print(f"  python evaluate.py --input_dir ./sample_test_data/input --output_dir ./sample_test_data/output")


if __name__ == '__main__':
    main()
