"""LPBA NIfTI loading, physical-vector export, and registration metrics."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import monai
import torch
from monai.metrics import compute_dice, compute_hausdorff_distance

LABEL_METRIC_DEFINITION = {
    "implementation": "MONAI", "version": monai.__version__,
    "dice": "compute_dice(ignore_empty=False); macro mean over union of nonzero labels",
    "hd95": "max of the two directed 95th-percentile Euclidean surface distances",
    "spacing": "millimetres in NIfTI array-axis order",
    "missing_label": "Dice=0; HD95=infinite represented by null plus hd95_infinite=true; aggregate invalid",
}


def write_json(path, data):
    """Atomic, strict JSON; unavailable measurements are represented by null."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_image(path):
    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.float32).copy()
    if data.ndim != 3 or min(data.shape) < 3 or not np.isfinite(data).all():
        raise ValueError(f"Expected a finite 3D MRI with dimensions >=3: {path}")
    foreground = data != 0
    if not foreground.any():
        raise ValueError(f"Empty MRI: {path}")
    low, high = np.percentile(data[foreground], [1, 99])
    if high <= low:
        raise ValueError(f"MRI has no usable intensity range: {path}")
    data = np.clip((data - low) / (high - low), 0, 1).astype(np.float32)
    data[~foreground] = 0
    return image, data, {"percentile_1": float(low), "percentile_99": float(high),
                         "method": "nonzero foreground percentile [1,99] -> [0,1]; background zero"}


def require_same_grid(reference, other, name):
    if reference.shape != other.shape or not np.allclose(reference.affine, other.affine, atol=1e-4):
        raise ValueError(f"{name} does not share the reference voxel grid/affine; resample it first")


def load_label(path, reference):
    image = nib.load(str(path))
    require_same_grid(reference, image, str(path))
    data = np.asarray(image.dataobj)
    if not np.isfinite(data).all() or not np.allclose(data, np.rint(data)):
        raise ValueError(f"Label map must contain finite integer IDs: {path}")
    return np.rint(data).astype(np.int32)


def save_volume(path, data, reference, vector=False):
    header = reference.header.copy()
    header.set_data_dtype(data.dtype)
    if vector:
        header.set_intent("vector")
    nib.save(nib.Nifti1Image(data, reference.affine, header), str(path))


def voxel_vector_to_world(displacement, affine):
    """Sampler channels (W,H,D) -> NIfTI array axes -> world RAS millimetres."""
    return (displacement[..., [2, 1, 0]] @ affine[:3, :3].T).astype(np.float32)


def jacobian_determinant(displacement):
    """det(I+du/dx); pixel-space determinant equals its physical conjugate."""
    rows = []
    for component in range(3):
        dz, dy, dx = np.gradient(displacement[..., component])
        rows.append([dx, dy, dz])
        rows[-1][component] = rows[-1][component] + 1
    a, b, c = rows
    return (a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])).astype(np.float32)


def image_metrics(first, second):
    mse = float(np.mean((first.astype(np.float64) - second) ** 2))
    a, b = first.astype(np.float64).ravel(), second.astype(np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {"mse": mse, "psnr_db": float(-10 * np.log10(mse)) if mse > 0 else None,
            "psnr_infinite": mse == 0,
            "global_ncc": float(np.dot(a, b) / denominator) if denominator > 0 else None}


def label_metrics(fixed, warped, spacing, compute_hd95=True):
    """MONAI labelwise Dice and symmetric HD95, excluding label 0.

    Each binary foreground class is passed as [B=1,C=1,D,H,W] with
    include_background=True so MONAI retains this foreground-only channel.
    Mean scores weight anatomical labels equally, not by their voxel counts.
    Strict JSON represents infinite HD95 by null plus an explicit flag.
    """
    fixed, warped = np.asarray(fixed), np.asarray(warped)
    spacing = np.asarray(spacing, dtype=float)
    if fixed.shape != warped.shape or fixed.ndim != 3 or not fixed.size:
        raise ValueError('Expected matching nonempty 3D label maps')
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Spacing must contain three positive finite values in array-axis order')
    labels = np.union1d(np.unique(fixed), np.unique(warped))
    per_label, missing, distances = {}, [], []
    for label in labels[labels != 0]:
        a, b = fixed == label, warped == label
        actual = torch.from_numpy(a)[None, None]
        prediction = torch.from_numpy(b)[None, None]
        dice = float(compute_dice(prediction, actual, include_background=True,
                                  ignore_empty=False).item())
        absent = not a.any() or not b.any()
        hd95 = None
        if absent:
            missing.append(int(label))
        elif compute_hd95:
            hd95 = float(compute_hausdorff_distance(prediction, actual,
                include_background=True, distance_metric="euclidean", percentile=95,
                directed=False, spacing=spacing.tolist()).item())
            if not np.isfinite(hd95):
                raise RuntimeError(f'Nonfinite MONAI HD95 for nonempty label {label}')
            distances.append(hd95)
        per_label[str(int(label))] = {"dice": dice, "hd95_mm": hd95,
                                     "hd95_infinite": bool(compute_hd95 and absent)}
    return {"dice": float(np.mean([m["dice"] for m in per_label.values()])) if per_label else None,
            "hd95_mm": float(np.mean(distances)) if distances and not missing else None,
            "hd95_finite_labels_mean_mm": float(np.mean(distances)) if distances else None,
            "hd95_infinite": bool(compute_hd95 and missing),
            "missing_labels": missing, "label_count": len(per_label),
            "metric_definition": {**LABEL_METRIC_DEFINITION, "hd95_enabled": bool(compute_hd95)}}, per_label
