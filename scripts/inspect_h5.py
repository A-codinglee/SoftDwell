#!/usr/bin/env python3
import h5py
import numpy as np

# ---- UPDATE THIS PATH ----
FILE_PATH = "/home/woody/mfpb/mfpb003h/time_series_datasets/COCO_SNR_0p1_0p2_100000_kij_100_1000000_samples_1000000_lvl_20000_22000_recStepResp_spectralBathNoise_grind_K24_range_10us_100ms_J24_2026_03_31.h5"
# --------------------------

RATE_POSITIONS = [
    (0, 1), (1, 0),
    (1, 2), (2, 1),
    (2, 3), (3, 2),
]
RATE_NAMES = ["k01","k10","k12","k21","k23","k32"]

def inspect_rates_and_snr(file_path: str, chunk: int = 4096):
    with h5py.File(file_path, "r") as f:
        Q = f["labels/Q"]              # (N,5,5)
        levels = f["labels/levels"]    # (N,5)
        sigma = f["labels/noise_sigma"]# (N,)

        N = Q.shape[0]

        # --- per-rate min/max ---
        per_min = np.full(len(RATE_POSITIONS), np.inf, dtype=np.float64)
        per_max = np.full(len(RATE_POSITIONS), -np.inf, dtype=np.float64)

        # --- pooled rate stats (optional) ---
        rate_global_min = np.inf
        rate_global_max = -np.inf

        # --- SNR min/max ---
        snr_min = np.inf
        snr_max = -np.inf

        for start in range(0, N, chunk):
            end = min(N, start + chunk)

            q = Q[start:end]               # (B,5,5)
            lv = levels[start:end]         # (B,5)
            sg = sigma[start:end]          # (B,)

            # Extract the 8 rates: (B,8)
            y = np.empty((q.shape[0], len(RATE_POSITIONS)), dtype=np.float64)
            for r, (i, j) in enumerate(RATE_POSITIONS):
                y[:, r] = q[:, i, j]

            # Per-rate min/max
            per_min = np.minimum(per_min, y.min(axis=0))
            per_max = np.maximum(per_max, y.max(axis=0))

            # Global min/max across all extracted rates
            rate_global_min = min(rate_global_min, float(y.min()))
            rate_global_max = max(rate_global_max, float(y.max()))

            # SNR = (max(level)-min(level)) / sigma
            span = lv.max(axis=1) - lv.min(axis=1)
            snr = span / np.maximum(sg, 1e-12)

            snr_min = min(snr_min, float(snr.min()))
            snr_max = max(snr_max, float(snr.max()))

        print("\n=== Rate ranges (extracted from labels/Q) ===")
        for name, mn, mx in zip(RATE_NAMES, per_min, per_max):
            print(f"{name:4s}: min={mn:.6g}  max={mx:.6g}")

        print(f"\n[rates pooled] min={rate_global_min:.6g}  max={rate_global_max:.6g}")

        # Log10 ranges (useful if you train on log10 rates)
        log_min = np.log10(max(rate_global_min, 1e-12))
        log_max = np.log10(max(rate_global_max, 1e-12))
        print(f"[rates pooled log10] min={log_min:.6g}  max={log_max:.6g}")

        print("\n=== SNR range (computed) ===")
        print(f"SNR min={snr_min:.6g}  max={snr_max:.6g}")

        # Sanity: also print the implied level span range and sigma range
        # (quickly from attributes already in memory via chunking is more work; optional)
        # If you want, I can add chunked span/sigma min/max too.

        
def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: '{name}'")
        print(f"   Shape: {obj.shape}")
        print(f"   Dtype: {obj.dtype}")
        
        # 1. Print string labels if applicable
        if len(obj.shape) == 1 and (obj.dtype.kind in 'SUO' or "name" in name.lower()):
            print("   First 3 values:")
            for i in range(min(3, obj.shape[0])):
                val = obj[i]
                if isinstance(val, (bytes, np.bytes_)):
                    val = val.decode('utf-8')
                print(f"     [{i}]: {val}")

        # 2. NUMERICAL DATA INSPECTION (Added feature)
        # If it's a large numerical array (likely your inputs/targets), verify ranges
        elif obj.shape and obj.dtype.kind in 'fi': # f=float, i=int
            # Load just a small chunk to keep it fast
            sample = obj[:1000] if obj.shape[0] > 1000 else obj[:] 
            
            print(f"   [Stats on first {len(sample)} samples]:")
            print(f"     Min: {np.min(sample):.4f} | Max: {np.max(sample):.4f}")
            print(f"     Mean: {np.mean(sample):.4f} | Std: {np.std(sample):.4f}")
            
            if np.issubdtype(obj.dtype, np.floating):
                if np.isnan(sample).any():
                    print("     [!!!] WARNING: NaNs detected in this chunk!")
                else:
                    print("     [OK] No NaNs detected in chunk.")
        
        print("-" * 30)

def main():
    print(f"Inspecting: {FILE_PATH}\n")
    with h5py.File(FILE_PATH, "r") as f:
        print(f"Keys found: {list(f.keys())}")
        print("=" * 30)
        f.visititems(print_structure)

    # Add this:
    inspect_rates_and_snr(FILE_PATH, chunk=4096)

if __name__ == "__main__":
    main()