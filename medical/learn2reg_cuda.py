"""CUDA implementations of the native Learn2Reg validation metrics."""
from functools import lru_cache

import numpy as np
import torch
import triton
import triton.language as tl
from scipy.ndimage import find_objects
import surface_distance
from surface_distance import lookup_tables


@triton.jit(do_not_specialize=['length'])
def _edt_axis(values, output, length, CANDIDATES: tl.constexpr, POINTS: tl.constexpr):
    """Squared 1D Euclidean distance transform of contiguous rows."""
    row = tl.program_id(0)
    x = tl.program_id(1) * POINTS + tl.arange(0, POINTS)
    p = tl.arange(0, CANDIDATES)
    cost = tl.load(values + row * length + p, p < length, other=1 << 28)
    delta = x[:, None] - p[None, :]
    nearest = tl.min(cost[None, :] + delta * delta, axis=1)
    tl.store(output + row * length + x, nearest, x < length)


def _squared_edt(surface):
    """Separable exact min of squared distances for unit voxel spacing."""
    cost = torch.full(surface.shape, 1 << 28, device=surface.device, dtype=torch.int32)
    cost.masked_fill_(surface, 0)
    for axis in (2, 1, 0):
        rows = cost.movedim(axis, -1).contiguous()
        length = rows.shape[-1]
        count = rows.numel() // length
        out = torch.empty_like(rows)
        _edt_axis[(count, triton.cdiv(length, 16))](
            rows, out, length, triton.next_power_of_2(length), 16)
        cost = out.movedim(-1, axis)
    return cost


def _surfel_codes(mask):
    # Each foreground voxel contributes to the eight adjacent corners, using
    # surface-distance's 2x2x2 neighbourhood encoding and its +1 padded extent.
    result = torch.zeros(tuple(n + 1 for n in mask.shape), device=mask.device, dtype=torch.uint8)
    mask = mask.to(torch.uint8)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                roi = tuple(slice(offset, offset + size) for offset, size in zip((dx, dy, dz), mask.shape))
                result[roi] |= mask << (dx * 4 + dy * 2 + dz)
    return result


@lru_cache(maxsize=1)
def _unit_surfel_areas():
    return lookup_tables.create_table_neighbour_code_to_surface_area(np.ones(3))


def _directed_surface_distances(codes, other_surface):
    surface = (codes != 0) & (codes != 255)
    squared = _squared_edt(other_surface)
    # Transfer only compact surface records. Use the published CPU percentile
    # on exactly the same ordering and FP64 cumulative areas, including ties.
    records = torch.stack((squared[surface], codes[surface].int()), dim=1).cpu().numpy()
    distances = np.sqrt(records[:, 0].astype(np.float64))
    areas = _unit_surfel_areas()[records[:, 1]]
    order = np.lexsort((areas, distances))
    return distances[order], areas[order]


def _hd95(mask, warped):
    a, b = _surfel_codes(mask), _surfel_codes(warped)
    ab, areas_a = _directed_surface_distances(a, (b != 0) & (b != 255))
    ba, areas_b = _directed_surface_distances(b, (a != 0) & (a != 255))
    return surface_distance.compute_robust_hausdorff(
        {'distances_gt_to_pred': ab, 'surfel_areas_gt': areas_a,
         'distances_pred_to_gt': ba, 'surfel_areas_pred': areas_b}, 95.0)


def _pullback(moving, mapping):
    # SciPy order=0, mode=constant. Half-integer ties round up and an input
    # outside the domain is zero even when its rounded index is inside.
    valid = torch.ones_like(mapping[..., 0], dtype=torch.bool)
    indices = []
    for dim, n in enumerate(moving.shape):
        coord = mapping[..., dim]
        valid &= (coord >= 0) & (coord <= n - 1)
        indices.append(torch.floor(coord.double() + 0.5).long().clamp_(0, n - 1))
    return torch.where(valid, moving[tuple(indices)], 0)


@torch.no_grad()
def label_metrics(fixed, moving, mapping, labels):
    """Compute scores for released integer label images and a native CUDA map."""
    # The CUDA context also selects the device for custom Triton kernels.
    with torch.cuda.device(mapping.device):
        fixed_gpu = torch.from_numpy(fixed).to(mapping.device)
        moving_gpu = torch.from_numpy(moving).to(mapping.device)
        warped = _pullback(moving_gpu, mapping)
        boxes = [find_objects(array, max_label=max(labels)) for array in
                 (fixed, moving, warped.cpu().numpy())]
        dice, hd95 = [], []
        for label in labels:
            truth, original, pred = (box[label - 1] for box in boxes)
            if truth is None or original is None:
                dice.append(np.nan)
                hd95.append(np.nan)
            elif pred is None:
                dice.append(0.)
                hd95.append(np.inf)
            else:
                roi = tuple(slice(min(a.start, b.start), max(a.stop, b.stop))
                            for a, b in zip(truth, pred))
                mask, pulled = fixed_gpu[roi] == label, warped[roi] == label
                intersection = torch.count_nonzero(mask & pulled)
                volume = torch.count_nonzero(mask) + torch.count_nonzero(pulled)
                dice.append((2 * intersection.double() / volume).item())
                hd95.append(_hd95(mask, pulled))
    return (np.nanmean(dice), dice), (np.nanmean(hd95), hd95)
