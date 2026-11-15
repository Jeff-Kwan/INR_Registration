"""Native-grid DIR-Lab folding counts without copying dense maps to the CPU."""
import torch
import triton
import triton.language as tl


@triton.jit
def _column(mapping, indices, coordinate, size: tl.constexpr, stride: tl.constexpr,
            xyz, component: tl.constexpr, axis: tl.constexpr, valid):
    # Reproduce np.gradient(displacement), including one-sided edge differences.
    before = (coordinate > 0).to(tl.int32)
    after = (coordinate + 1 < size).to(tl.int32)
    lo = tl.load(mapping + 3 * (indices - before * stride) + component, valid, 0)
    hi = tl.load(mapping + 3 * (indices + after * stride) + component, valid, 0)
    if component == axis:
        lo = lo - (xyz - before)
        hi = hi - (xyz + after)
    else:
        lo = lo - xyz
        hi = hi - xyz
    difference = (hi - lo) * tl.where((before + after) == 2, 0.5, 1.0)
    if component == axis:
        difference += 1.0
    return difference


@triton.jit
def _folding(mapping, mask, counts, D: tl.constexpr, H: tl.constexpr,
             W: tl.constexpr, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    # Use wide offsets only when the interleaved vector field needs them.
    index_block = block.to(tl.int64) if 3 * D * H * W >= 2**31 else block
    i = index_block * BLOCK + tl.arange(0, BLOCK)
    valid = i < D * H * W
    x, y, z = i % W, (i // W) % H, i // (W * H)
    a = _column(mapping, i, x, W, 1, x, 0, 0, valid)
    b = _column(mapping, i, y, H, W, x, 0, 1, valid)
    c = _column(mapping, i, z, D, W * H, x, 0, 2, valid)
    d = _column(mapping, i, x, W, 1, y, 1, 0, valid)
    e = _column(mapping, i, y, H, W, y, 1, 1, valid)
    f = _column(mapping, i, z, D, W * H, y, 1, 2, valid)
    g = _column(mapping, i, x, W, 1, z, 2, 0, valid)
    h = _column(mapping, i, y, H, W, z, 2, 1, valid)
    j = _column(mapping, i, z, D, W * H, z, 2, 2, valid)
    det = a * (e*j - f*h) - b * (d*j - f*g) + c * (d*h - e*g)
    folded = (det <= 0) & valid
    lung = tl.load(mask + i, valid, 0) != 0
    tl.store(counts + block * 3, tl.sum(folded.to(tl.int32), 0))
    tl.store(counts + block * 3 + 1, tl.sum((folded & lung).to(tl.int32), 0))
    tl.store(counts + block * 3 + 2, tl.sum(lung.to(tl.int32), 0))


def folding_fractions(mapping, mask):
    shape = mask.shape
    if (mapping.shape != (*shape, 3) or len(shape) != 3 or min(shape) < 2
            or mapping.device != mask.device or mapping.dtype != torch.float32):
        raise ValueError('Expected a native float32 3D map and matching mask')
    blocks = triton.cdiv(mask.numel(), 256)
    counts = torch.empty((blocks, 3), device=mask.device, dtype=torch.int32)
    with torch.cuda.device(mapping.device):
        _folding[(blocks,)](mapping.contiguous(), mask.contiguous(), counts, *shape,
                            BLOCK=256, enable_fp_fusion=False)
    whole, lung, lung_size = counts.sum(0).tolist()
    return whole / mask.numel(), lung / lung_size
