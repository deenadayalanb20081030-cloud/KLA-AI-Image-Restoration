"""
================================================================================
  KLA AI Hackathon: Standalone .NPY -> .PNG Conversion & Visualization Module
================================================================================
  Purpose:
    Dedicated utility to convert raw floating-point .npy wafer arrays to 
    high-fidelity .png images for visual inspection by evaluators and metrology engineers.

  Features:
    • Multi-threaded batch conversion for 1,000+ to 10,000+ files
    • Multiple tone-mapping modes:
        - 'standard': Physical [0.0, 1.0] intensity projection with clipping
        - 'percentile': 1% - 99% robust contrast stretching for faint defect review
        - 'minmax': Dynamic per-array min-max normalization
        - 'colormap': False-color visualization (Viridis, Inferno, Turbo, Jet)
    • Unclipped speckle dynamics preservation with NaN/Inf sanitization
    • Clean Python API and command-line interface (CLI)

  Usage Examples:
    # 1. Batch directory conversion
    python convert_npy_to_png.py --input_dir ./outputs --output_dir ./outputs_png

    # 2. Single file conversion
    python convert_npy_to_png.py --input_file ./sample.npy --output_file ./sample.png

    # 3. False-color defect review
    python convert_npy_to_png.py --input_dir ./outputs --output_dir ./outputs_inferno --colormap inferno

    # 4. Python module import
    from convert_npy_to_png import npy_to_png, batch_convert_npy_to_png
    png_img = npy_to_png("wafer_tile.npy", mode="standard")
================================================================================
"""

import os
import sys
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def npy_array_to_uint8(arr: np.ndarray, mode: str = "standard", colormap: str = "none") -> np.ndarray:
    """
    Converts a NumPy array of arbitrary dimension/dtype to an 8-bit (uint8) image array.
    
    Parameters:
      arr (np.ndarray): Input NumPy array (2D or 3D).
      mode (str): Normalization mode ('standard', 'percentile', 'minmax').
      colormap (str): Colormap ('none', 'gray', 'viridis', 'inferno', 'turbo', 'jet').

    Returns:
      np.ndarray (uint8): 2D [H, W] grayscale or 3D [H, W, 3] RGB image.
    """
    # 1. Sanitize NaN and Inf values
    if not np.issubdtype(arr.dtype, np.floating) and not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32)
    
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

    # 2. Squeeze extra dimensions
    arr = np.squeeze(arr)

    # 3. Channel adaptation for 3D arrays
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3):  # CHW format -> HWC
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        elif arr.shape[2] == 3 and colormap == "none":
            # Convert RGB to Grayscale for standard metrology display
            arr = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]

    # If already uint8 and standard mode, return directly
    if arr.dtype == np.uint8 and mode == "standard" and colormap == "none":
        return arr

    arr_float = arr.astype(np.float32)

    # 4. Normalization Modes
    if mode == "percentile":
        # 1st - 99th percentile contrast stretch for faint defect discovery
        p_low, p_high = np.percentile(arr_float, (1.0, 99.0))
        if p_high > p_low:
            norm = (arr_float - p_low) / (p_high - p_low)
        else:
            norm = arr_float
        norm = np.clip(norm, 0.0, 1.0)

    elif mode == "minmax":
        # Full dynamic range min-max scaling
        vmin, vmax = np.min(arr_float), np.max(arr_float)
        if vmax > vmin:
            norm = (arr_float - vmin) / (vmax - vmin)
        else:
            norm = np.zeros_like(arr_float)
        norm = np.clip(norm, 0.0, 1.0)

    else:  # "standard"
        # Standard nominal [0.0, 1.0] physical projection
        if np.max(arr_float) > 2.0:  # Array was in [0, 255]
            arr_float = arr_float / 255.0
        norm = np.clip(arr_float, 0.0, 1.0)

    img_uint8 = (norm * 255.0).round().astype(np.uint8)

    # 5. Optional False-Color Colormaps for Metrology Visualization
    if colormap.lower() not in ("none", "gray", "grayscale"):
        if HAS_CV2:
            cmap_dict = {
                "viridis": cv2.COLORMAP_VIRIDIS if hasattr(cv2, "COLORMAP_VIRIDIS") else cv2.COLORMAP_JET,
                "inferno": cv2.COLORMAP_INFERNO if hasattr(cv2, "COLORMAP_INFERNO") else cv2.COLORMAP_HOT,
                "turbo": cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_RAINBOW,
                "jet": cv2.COLORMAP_JET,
                "hot": cv2.COLORMAP_HOT
            }
            cmap_code = cmap_dict.get(colormap.lower(), cv2.COLORMAP_JET)
            img_bgr = cv2.applyColorMap(img_uint8, cmap_code)
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            # Fallback simple heat colormap in pure NumPy
            r = np.clip(img_uint8 * 2, 0, 255).astype(np.uint8)
            g = np.clip(255 - np.abs(img_uint8.astype(int) - 128) * 2, 0, 255).astype(np.uint8)
            b = np.clip((255 - img_uint8) * 2, 0, 255).astype(np.uint8)
            return np.stack([r, g, b], axis=-1)

    return img_uint8


def npy_to_png(npy_path: str, png_path: str = None, mode: str = "standard", colormap: str = "none") -> Image.Image:
    """
    Converts a single .npy file to a .png image file.
    
    Parameters:
      npy_path (str): Path to input .npy file.
      png_path (str, optional): Destination .png path. If None, saves alongside .npy.
      mode (str): Normalization mode ('standard', 'percentile', 'minmax').
      colormap (str): Optional colormap ('none', 'viridis', 'inferno', 'turbo').

    Returns:
      PIL.Image: The generated PIL Image object.
    """
    arr = np.load(npy_path)
    img_uint8 = npy_array_to_uint8(arr, mode=mode, colormap=colormap)
    pil_img = Image.fromarray(img_uint8)

    if png_path is None:
        png_path = str(Path(npy_path).with_suffix('.png'))
    
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    pil_img.save(png_path)
    return pil_img


def _convert_worker(task: tuple) -> tuple:
    npy_path, out_png_path, mode, colormap = task
    try:
        arr = np.load(npy_path)
        img_uint8 = npy_array_to_uint8(arr, mode=mode, colormap=colormap)
        Image.fromarray(img_uint8).save(out_png_path)
        return (True, npy_path, out_png_path, None)
    except Exception as e:
        return (False, npy_path, out_png_path, str(e))


def batch_convert_npy_to_png(
    input_dir: str,
    output_dir: str,
    mode: str = "standard",
    colormap: str = "none",
    workers: int = 8
) -> dict:
    """
    Multi-threaded batch converter for large directories of .npy wafer arrays.
    
    Parameters:
      input_dir (str): Directory containing .npy files.
      output_dir (str): Destination directory for .png images.
      mode (str): 'standard', 'percentile', or 'minmax'.
      colormap (str): 'none', 'viridis', 'inferno', 'turbo', 'jet'.
      workers (int): Number of parallel threads.

    Returns:
      dict: Summary statistics (total, success, failed, time_sec, fps).
    """
    t_start = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)

    npy_files = list(Path(input_dir).rglob("*.npy"))
    if not npy_files:
        print(f"[WARNING] No .npy files found in {input_dir}")
        return {"total": 0, "success": 0, "failed": 0, "time_sec": 0, "fps": 0}

    tasks = []
    for p in npy_files:
        rel_path = p.relative_to(input_dir)
        out_png = Path(output_dir) / rel_path.with_suffix('.png')
        os.makedirs(out_png.parent, exist_ok=True)
        tasks.append((str(p), str(out_png), mode, colormap))

    total = len(tasks)
    success = 0
    failed = 0

    print(f"\n[INFO] Converting {total} .npy files -> .png (Mode: {mode}, Colormap: {colormap}, Threads: {workers})...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_convert_worker, t) for t in tasks]
        for f in as_completed(futures):
            ok, src, dst, err = f.result()
            if ok:
                success += 1
            else:
                failed += 1
                print(f"[ERROR] Failed converting {src}: {err}")

    t_elapsed = max(time.perf_counter() - t_start, 1e-4)
    fps = total / t_elapsed

    print(f"[OK] Completed: {success}/{total} converted in {t_elapsed:.2f}s ({fps:.1f} files/sec).")
    return {
        "total": total,
        "success": success,
        "failed": failed,
        "time_sec": t_elapsed,
        "fps": fps
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="KLA Metrology .NPY -> .PNG High-Speed Conversion & Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_npy_to_png.py --input_dir ./outputs --output_dir ./outputs_png
  python convert_npy_to_png.py --input_file ./outputs/sample_0001.npy --output_file ./sample_0001.png
  python convert_npy_to_png.py --input_dir ./outputs --output_dir ./outputs_inferno --colormap inferno
        """
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--input_dir", type=str, help="Directory containing .npy files to convert")
    group.add_argument("-f", "--input_file", type=str, help="Single .npy file to convert")

    parser.add_argument("-o", "--output_dir", type=str, default="./outputs_png",
                        help="Directory to save output .png images (for batch mode)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Destination .png file path (for single file mode)")
    parser.add_argument("--mode", type=str, choices=["standard", "percentile", "minmax"], default="standard",
                        help="Tone mapping mode: 'standard' (0-1), 'percentile' (1-99%%), 'minmax'")
    parser.add_argument("--colormap", type=str, choices=["none", "viridis", "inferno", "turbo", "jet", "hot"],
                        default="none", help="False-color colormap for visual metrology inspection")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads for batch conversion")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.input_file:
        if not os.path.isfile(args.input_file):
            print(f"[ERROR] Input file does not exist: {args.input_file}")
            sys.exit(1)
        
        out_path = args.output_file or str(Path(args.input_file).with_suffix('.png'))
        npy_to_png(args.input_file, out_path, mode=args.mode, colormap=args.colormap)
        print(f"[OK] Converted single file:\n  Source: {args.input_file}\n  Output: {out_path}")
    
    elif args.input_dir:
        if not os.path.isdir(args.input_dir):
            print(f"[ERROR] Input directory does not exist: {args.input_dir}")
            sys.exit(1)
        
        batch_convert_npy_to_png(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            colormap=args.colormap,
            workers=args.workers
        )


if __name__ == "__main__":
    main()
