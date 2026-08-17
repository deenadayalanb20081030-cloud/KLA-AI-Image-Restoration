"""
KLA AI Hackathon: High-Scale Semiconductor Metrology Dataset Generator (1,000+ Samples)
Synthesizes paired degraded and ground truth wafer images across 8 major semiconductor fab categories:
1. Logic FinFET / Gate-All-Around (GAA) standard cells
2. 3D NAND Flash Memory vertical channel hole arrays
3. DRAM High-Density Capacitor Trenches & Bitlines
4. Advanced Packaging Through-Silicon Vias (TSVs) & Microbumps
5. EUV Optical Overlay & Diffraction Gratings
6. CMP Polishing Scratches & Particulate Defects
7. SEM Crystal Dendrite & Dislocation Networks
8. Out-of-Distribution (OOD) Anisotropic Multi-Modal Samples

Outputs both standard .png images and raw float32 .npy arrays.
"""

import os
import argparse
import time
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


def synthesize_logic_finfet(h=512, w=512):
    """Logic FinFET / GAA standard cell routing lines & contact pads"""
    arr = np.full((h, w), 0.15, dtype=np.float32)
    # Horizontal fin lines
    pitch = 16
    for y in range(0, h, pitch):
        arr[y:min(h, y+6), :] = 0.75
    # Vertical gate electrodes
    for x in range(0, w, 32):
        arr[:, x:min(w, x+10)] = 0.90
    # Contact via plugs
    for y in range(8, h, 32):
        for x in range(16, w, 32):
            y_min, y_max = max(0, y-3), min(h, y+3)
            x_min, x_max = max(0, x-3), min(w, x+3)
            arr[y_min:y_max, x_min:x_max] = 0.98
    return arr


def synthesize_3d_nand(h=512, w=512):
    """3D NAND vertical memory channel hole arrays & staircase wordlines"""
    arr = np.full((h, w), 0.20, dtype=np.float32)
    r = 6
    spacing_x = 24
    spacing_y = 20
    for row, y in enumerate(range(spacing_y, h - spacing_y, spacing_y)):
        offset_x = (spacing_x // 2) if (row % 2 == 1) else 0
        for x in range(spacing_x + offset_x, w - spacing_x, spacing_x):
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    if dx*dx + dy*dy <= r*r:
                        py, px = y + dy, x + dx
                        if 0 <= py < h and 0 <= px < w:
                            arr[py, px] = 0.85
    return arr


def synthesize_dram_trench(h=512, w=512):
    """DRAM capacitor trench arrays & orthogonal bitlines"""
    arr = np.full((h, w), 0.18, dtype=np.float32)
    for y in range(0, h, 12):
        arr[y:min(h, y+4), :] = 0.70
    for x in range(0, w, 20):
        arr[:, x:min(w, x+6)] = 0.82
    return arr


def synthesize_tsv_packaging(h=512, w=512):
    """Advanced Packaging Through-Silicon Vias (TSVs) & C4 Microbumps"""
    arr = np.full((h, w), 0.10, dtype=np.float32)
    r = max(12, h // 20)
    spacing = r * 3
    for y in range(spacing // 2, h, spacing):
        for x in range(spacing // 2, w, spacing):
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist <= r:
                        py, px = y + dy, x + dx
                        if 0 <= py < h and 0 <= px < w:
                            arr[py, px] = 0.95 - (dist / r) * 0.45
    return arr


def synthesize_euv_grating(h=512, w=512):
    """EUV Optical Diffraction Gratings & Overlay Metrology Targets"""
    y, x = np.mgrid[0:h, 0:w]
    grating = np.sin(x * 2 * np.pi / 20.0)
    arr = np.where(grating > 0, 0.88, 0.22).astype(np.float32)
    # Overlay frame
    f_min, f_max = int(h * 0.2), int(h * 0.8)
    t = max(4, h // 50)
    arr[f_min:f_min+t, f_min:f_max] = 0.95
    arr[f_max-t:f_max, f_min:f_max] = 0.95
    arr[f_min:f_max, f_min:f_min+t] = 0.95
    arr[f_min:f_max, f_max-t:f_max] = 0.95
    return arr


def synthesize_cmp_scratches(h=512, w=512):
    """Chemical Mechanical Planarization (CMP) surface micro-scratches & slurry particles"""
    arr = np.full((h, w), 0.50, dtype=np.float32)
    # Linear polishing scratches
    for i in range(10):
        slope = np.random.uniform(-0.5, 0.5)
        intercept = np.random.uniform(20, h - 20)
        for x in range(w):
            y = int(slope * x + intercept)
            if 0 <= y < h:
                arr[max(0, y-1):min(h, y+2), x] = np.random.choice([0.15, 0.92])
    # Particulate contamination
    for _ in range(25):
        px, py = np.random.randint(10, w-10), np.random.randint(10, h-10)
        arr[max(0, py-2):min(h, py+3), max(0, px-2):min(w, px+3)] = 0.98
    return arr


def synthesize_dendrite_crystal(h=512, w=512):
    """SEM Dendrite Crystal Dislocation Defect"""
    arr = np.full((h, w), 0.05, dtype=np.float32)
    cx, cy = w // 2, h // 2
    max_len = int(min(h, w) * 0.45)
    for i in range(16):
        angle = (i / 16.0) * np.pi * 2 + np.random.uniform(-0.1, 0.1)
        length = np.random.randint(int(max_len * 0.6), max_len)
        for r in range(10, length):
            px = int(cx + r * np.cos(angle) + np.sin(r * 0.15) * 5)
            py = int(cy + r * np.sin(angle) + np.cos(r * 0.15) * 5)
            if 0 <= px < w and 0 <= py < h:
                arr[py, px] = 0.85
                if r % 3 == 0:
                    for branch in [-0.5, 0.5]:
                        bx = int(px + np.cos(angle + branch) * 12)
                        by = int(py + np.sin(angle + branch) * 12)
                        if 0 <= bx < w and 0 <= by < h:
                            arr[by, bx] = 0.70
    return arr


def synthesize_ood_complex(h=512, w=512):
    """Out-of-Distribution Anisotropic Multi-Modal Texture"""
    y, x = np.mgrid[0:h, 0:w]
    u = x / w
    v = y / h
    n = np.sin(u * 60 + np.sin(v * 45)) * 0.25
    n += np.cos(v * 80 + np.cos(u * 35)) * 0.25
    arr = np.clip(0.5 + n, 0.05, 0.95).astype(np.float32)
    return arr


MODALITY_GENERATORS = [
    ("logic_finfet", synthesize_logic_finfet),
    ("nand_3d", synthesize_3d_nand),
    ("dram_trench", synthesize_dram_trench),
    ("tsv_packaging", synthesize_tsv_packaging),
    ("euv_grating", synthesize_euv_grating),
    ("cmp_scratches", synthesize_cmp_scratches),
    ("dendrite_defect", synthesize_dendrite_crystal),
    ("ood_metrology", synthesize_ood_complex),
]


def apply_physical_degradation(gt_arr: np.ndarray, downscale: int = 2) -> np.ndarray:
    """
    Applies realistic multi-scale semiconductor degradation:
    1. Multiplicative Gamma speckle noise (>1.0 intensity spread)
    2. Additive Gaussian thermal sensor noise
    3. Poisson shot noise
    4. 2x Spatial resolution downsampling (512x512 -> 256x256 or 256x256 -> 128x128)
    """
    h, w = gt_arr.shape
    # 1. Multiplicative Gamma Speckle
    k = np.random.uniform(8.0, 14.0)
    theta = 1.0 / k
    speckle = np.random.gamma(shape=k, scale=theta, size=(h, w))
    noisy = gt_arr * speckle

    # 2. Gaussian Thermal Noise
    sigma = np.random.uniform(0.008, 0.035)
    noisy = noisy + np.random.normal(0, sigma, size=(h, w))

    # 3. Poisson Shot Noise
    if np.random.rand() < 0.35:
        peak = np.random.uniform(80.0, 200.0)
        noisy = np.random.poisson(np.clip(noisy, 0, None) * peak) / peak

    # 4. Spatial Downsampling via PIL Bicubic
    pil_img = Image.fromarray(np.clip(noisy * 255.0, 0, 255).astype(np.uint8))
    pil_lr = pil_img.resize((w // downscale, h // downscale), resample=Image.Resampling.BICUBIC)
    lr_arr = np.array(pil_lr, dtype=np.float32) / 255.0

    return lr_arr.astype(np.float32)


def generate_single_sample(args_tuple):
    idx, modality_name, generator_fn, out_dir_input, out_dir_gt, resolution, save_npy = args_tuple
    h = w = resolution
    gt_arr = generator_fn(h, w)
    lr_arr = apply_physical_degradation(gt_arr, downscale=2)

    base_name = f"wafer_{idx:05d}_{modality_name}_{w//2}x{h//2}"

    # 1. Save Degraded Input Image (PNG & optional NPY)
    lr_png_path = os.path.join(out_dir_input, f"{base_name}.png")
    Image.fromarray(np.clip(lr_arr * 255.0, 0, 255).astype(np.uint8)).save(lr_png_path)

    # 2. Save Clean Ground Truth Image (PNG & optional NPY)
    gt_png_path = os.path.join(out_dir_gt, f"{base_name}.png")
    Image.fromarray(np.clip(gt_arr * 255.0, 0, 255).astype(np.uint8)).save(gt_png_path)

    if save_npy:
        np.save(os.path.join(out_dir_input, f"{base_name}.npy"), lr_arr)
        np.save(os.path.join(out_dir_gt, f"{base_name}.npy"), gt_arr)

    return idx


def main():
    parser = argparse.ArgumentParser(description="Generate 1,000+ Paired Semiconductor Wafer Samples for KLA Hackathon")
    parser.add_argument("--count", type=int, default=1000, help="Total number of paired images to generate (default: 1000)")
    parser.add_argument("--input_dir", type=str, default="./sample_test_data/input_1000", help="Directory for degraded LR inputs")
    parser.add_argument("--gt_dir", type=str, default="./sample_test_data/gt_1000", help="Directory for clean HR ground truth")
    parser.add_argument("--save_npy", action="store_true", default=True, help="Save raw float32 .npy arrays alongside .png images")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker threads")
    args = parser.parse_args()

    os.makedirs(args.input_dir, exist_ok=True)
    os.makedirs(args.gt_dir, exist_ok=True)

    print("=" * 80)
    print(f"  KLA AI Hackathon - High-Scale Dataset Generator (Target: {args.count} Paired Images)")
    print(f"  Modality Coverage: 8 Major Semiconductor Fab Categories")
    print(f"  Formats:           PNG Images (8-bit) + Raw Float32 .npy Arrays")
    print(f"  Input Directory:   {os.path.abspath(args.input_dir)}")
    print(f"  Ground Truth:      {os.path.abspath(args.gt_dir)}")
    print("=" * 80)

    t0 = time.perf_counter()
    tasks = []
    num_modalities = len(MODALITY_GENERATORS)

    for i in range(args.count):
        mod_name, gen_fn = MODALITY_GENERATORS[i % num_modalities]
        # Mix resolutions: 512x512 GT (downscaled to 256x256) and 256x256 GT (downscaled to 128x128)
        res = 512 if (i % 2 == 0) else 256
        tasks.append((i + 1, mod_name, gen_fn, args.input_dir, args.gt_dir, res, args.save_npy))

    print(f"[INFO] Synthesizing {args.count} paired samples using ThreadPool with {args.workers} workers...")

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for idx in executor.map(generate_single_sample, tasks):
            completed += 1
            if completed % 100 == 0 or completed == args.count:
                print(f"  -> Generated [{completed:04d}/{args.count:04d}] paired wafer tiles ({(completed/args.count)*100:.1f}%)")

    elapsed = time.perf_counter() - t0
    rate = args.count / max(elapsed, 1e-6)

    print("=" * 80)
    print(f"  [DATASET GENERATION COMPLETE]")
    print(f"  Total Paired Samples: {args.count} Degraded Inputs + {args.count} Clean Ground Truth")
    print(f"  Total Files Created:  {args.count * 4 if args.save_npy else args.count * 2} files (.png + .npy)")
    print(f"  Total Elapsed Time:   {elapsed:.2f} s ({rate:.1f} tiles / sec)")
    print(f"  Degraded Input Path:  {os.path.abspath(args.input_dir)}")
    print(f"  Ground Truth Path:    {os.path.abspath(args.gt_dir)}")
    print("=" * 80)


if __name__ == '__main__':
    main()
