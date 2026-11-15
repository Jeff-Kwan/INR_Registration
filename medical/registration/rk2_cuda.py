"""Frozen FP32 RK2 on native 3D grids. Each thread follows one voxel trajectory."""
import torch
import triton
import triton.language as tl


@triton.jit
def _pixel(pos, size: tl.constexpr):
    # Preserve the same align_corners normalization round trip as grid_sample.
    norm = tl.div_rn(2.0 * pos, size - 1.0) - 1.0
    return tl.minimum(tl.maximum((norm + 1.0) * 0.5 * (size - 1), 0.0), size - 1.0)


@triton.jit
def _sample(field, x, y, z, valid, D: tl.constexpr, H: tl.constexpr, W: tl.constexpr):
    px, py, pz = _pixel(x, W), _pixel(y, H), _pixel(z, D)
    ix, iy, iz = px.to(tl.int32), py.to(tl.int32), pz.to(tl.int32)
    vx = tl.full((x.shape[0],), 0, tl.float32)
    vy = tl.full((x.shape[0],), 0, tl.float32)
    vz = tl.full((x.shape[0],), 0, tl.float32)
    for dz in tl.static_range(2):
        for dy in tl.static_range(2):
            for dx in tl.static_range(2):
                nx, ny, nz = ix + dx, iy + dy, iz + dz
                wx = px - ix if dx else ix + 1 - px
                wy = py - iy if dy else iy + 1 - py
                wz = pz - iz if dz else iz + 1 - pz
                weight = wx * wy * wz
                at = ((nz * H + ny) * W + nx) * 3
                use = valid & (nx < W) & (ny < H) & (nz < D)
                vx = tl.fma(weight, tl.load(field + at, use, 0), vx)
                vy = tl.fma(weight, tl.load(field + at + 1, use, 0), vy)
                vz = tl.fma(weight, tl.load(field + at + 2, use, 0), vz)
    return vx, vy, vz


@triton.jit
def _integrate(mapping, velocity, output, step, substeps,
               D: tl.constexpr, H: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < D * H * W
    x = tl.load(mapping + 3*i, valid, 0)
    y = tl.load(mapping + 3*i + 1, valid, 0)
    z = tl.load(mapping + 3*i + 2, valid, 0)
    for _ in range(substeps):
        vx, vy, vz = _sample(velocity, x, y, z, valid, D, H, W)
        mx, my, mz = x + (0.5*step)*vx, y + (0.5*step)*vy, z + (0.5*step)*vz
        vx, vy, vz = _sample(velocity, mx, my, mz, valid, D, H, W)
        x, y, z = x + step*vx, y + step*vy, z + step*vz
    tl.store(output + 3*i, x, valid)
    tl.store(output + 3*i + 1, y, valid)
    tl.store(output + 3*i + 2, z, valid)


def integrate(mapping, velocity, dt, substeps):
    shape = mapping.shape[:3]
    out = torch.empty_like(mapping, memory_format=torch.contiguous_format)
    with torch.cuda.device(mapping.device):
        _integrate[(triton.cdiv(out.numel() // 3, 128),)](
            mapping.contiguous(), velocity.contiguous(), out, dt / substeps, substeps,
            *shape, BLOCK=128, enable_fp_fusion=False)
    return out
