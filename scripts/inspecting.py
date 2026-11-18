import h5py
import numpy as np

path = "/home/woody/mfpb/mfpb003h/time_series_datasets/COCO_SNR_5_100000_kij_fixed_fordemo_samples_1000000_lvl_20000_22000_recStepResp_spectralBathNoise_time_series_unclamped_2026_04_16.h5"

def fmt_attrs(obj):
    if not obj.attrs:
        return ""
    parts = []
    for k in obj.attrs.keys():
        v = obj.attrs[k]
        # keep it readable
        if isinstance(v, (bytes, str)):
            parts.append(f"{k}={v!r}")
        else:
            v_np = np.array(v)
            if v_np.size <= 8:
                parts.append(f"{k}={v_np.tolist()}")
            else:
                parts.append(f"{k}=array(shape={v_np.shape}, dtype={v_np.dtype})")
    return " | attrs: " + ", ".join(parts)

with h5py.File(path, "r") as f:
    print("ROOT keys:", list(f.keys()))
    print("ROOT attrs:", dict(f.attrs))

    def visitor(name, obj):
        indent = "  " * (name.count("/") + 1)
        if isinstance(obj, h5py.Group):
            print(f"{indent}[G] {name}{fmt_attrs(obj)}")
        elif isinstance(obj, h5py.Dataset):
            # dataset metadata only (no data read)
            print(
                f"{indent}[D] {name} shape={obj.shape} dtype={obj.dtype} "
                f"chunks={obj.chunks} compression={obj.compression}{fmt_attrs(obj)}"
            )

    print("\n--- TREE ---")
    f.visititems(visitor)
