"""Fused CUDA steps for the frozen 3D float32 brightness-flow PCG solve."""
import torch
import triton
import triton.language as tl


@triton.jit
def _prepare(source, gradient, inverse, n: tl.constexpr, width: tl.constexpr,
             height: tl.constexpr, wz, wy, wx, smoothness, ridge, block: tl.constexpr):
    i = tl.program_id(0) * block + tl.arange(0, block)
    valid = i < n
    x = i % width
    y = i // width % height
    plane: tl.constexpr = width * height
    z = i // plane
    degree = wz * ((z > 0).to(tl.float32) + (z < n // plane - 1).to(tl.float32))
    degree += wy * ((y > 0).to(tl.float32) + (y < height - 1).to(tl.float32))
    degree += wx * ((x > 0).to(tl.float32) + (x < width - 1).to(tl.float32))
    for component in tl.static_range(3):
        g = tl.load(source + i * 3 + component, valid, 0)
        j = component * n + i
        tl.store(gradient + j, g, valid)
        tl.store(inverse + j, 1. / (g * g + smoothness * degree + ridge), valid)


@triton.jit
def _apply(direction, gradient, applied, partial, n: tl.constexpr,
           width: tl.constexpr, height: tl.constexpr, wz, wy, wx,
           smoothness, ridge, block: tl.constexpr):
    # A contiguous block per component makes adjacent threads visit adjacent voxels.
    tile = tl.program_id(0)
    i = tile * block + tl.arange(0, block)
    valid = i < n
    x = i % width
    y = i // width % height
    plane: tl.constexpr = width * height
    z = i // plane
    projected = tl.full((block,), 0, tl.float32)
    for component in tl.static_range(3):
        j = component * n + i
        d = tl.load(direction + j, valid, 0)
        g = tl.load(gradient + j, valid, 0)
        projected += d * g
    product = tl.full((block,), 0, tl.float32)
    for component in tl.static_range(3):
        j = component * n + i
        p = direction + j
        d = tl.load(p, valid, 0)
        # Neumann boundaries replicate the center, including singleton axes.
        dz = ((d - tl.load(p - tl.where(z > 0, plane, 0), valid, 0))
              + (d - tl.load(p + tl.where(z < n // plane - 1, plane, 0), valid, 0)))
        dy = ((d - tl.load(p - tl.where(y > 0, width, 0), valid, 0))
              + (d - tl.load(p + tl.where(y < height - 1, width, 0), valid, 0)))
        dx = ((d - tl.load(p - tl.where(x > 0, 1, 0), valid, 0))
              + (d - tl.load(p + tl.where(x < width - 1, 1, 0), valid, 0)))
        g = tl.load(gradient + j, valid, 0)
        ad = g * projected + smoothness * (wz * dz + wy * dy + wx * dx) + ridge * d
        tl.store(applied + j, ad, valid)
        product += d * ad
    tl.store(partial + tile, tl.sum(product, axis=0))


@triton.jit
def _update(residual, applied, inverse, alpha, partial,
            length: tl.constexpr, block: tl.constexpr):
    tile = tl.program_id(0)
    i = tile * block + tl.arange(0, block)
    valid = i < length
    scale = tl.load(alpha)
    r = tl.load(residual + i, valid, 0) - scale * tl.load(applied + i, valid, 0)
    m = tl.load(inverse + i, valid, 0)
    tl.store(residual + i, r, valid)
    tl.store(partial + tile, tl.sum(r * (r * m), axis=0))


@triton.jit
def _direction(velocity, direction, residual, inverse, alpha, beta, length: tl.constexpr, block: tl.constexpr):
    i = tl.program_id(0) * block + tl.arange(0, block)
    valid = i < length
    r = tl.load(residual + i, valid, 0)
    z = r * tl.load(inverse + i, valid, 0)
    d = tl.load(direction + i, valid, 0)
    v = tl.load(velocity + i, valid, 0)
    tl.store(velocity + i, v + tl.load(alpha) * d, valid)
    tl.store(direction + i, z + tl.load(beta) * d, valid)


def planar(value):
    return value.movedim(-1, 0).contiguous().movedim(0, -1)


class CudaPcgStep:
    """Reuse scratch throughout one solve. Vector views are planar (C,Z,Y,X)."""

    def __init__(self, gradient, smoothness, ridge, weights):
        self.gradient = gradient.new_empty((3, *gradient.shape[:-1])).movedim(0, -1)
        self.inverse = torch.empty_like(self.gradient)
        self.applied = torch.empty_like(self.gradient)
        self.weights = weights
        self.smoothness = smoothness
        self.ridge = ridge
        depth, self.height, self.width, _ = gradient.shape
        self.n = depth * self.height * self.width
        self.apply_parts = gradient.new_empty(triton.cdiv(self.n, 256))
        self.update_parts = gradient.new_empty(triton.cdiv(3 * self.n, 512))
        with torch.cuda.device(gradient.device):
            _prepare[(self.apply_parts.numel(),)](
                gradient.contiguous(), self.gradient, self.inverse, self.n, self.width,
                self.height, *weights, smoothness, ridge, 256, enable_fp_fusion=False)

    def __call__(self, velocity, residual, direction, rz):
        with torch.cuda.device(self.gradient.device):
            _apply[(self.apply_parts.numel(),)](direction, self.gradient, self.applied,
                self.apply_parts, self.n, self.width, self.height, *self.weights,
                self.smoothness, self.ridge, 256, enable_fp_fusion=False)
            alpha = rz / self.apply_parts.sum().clamp_min(1e-30)
            _update[(self.update_parts.numel(),)](residual, self.applied,
                self.inverse, alpha, self.update_parts, 3*self.n, 512, enable_fp_fusion=False)
            next_rz = self.update_parts.sum()
            beta = next_rz / rz.clamp_min(1e-30)
            _direction[(self.update_parts.numel(),)](velocity, direction, residual, self.inverse, alpha, beta,
                3*self.n, 512, enable_fp_fusion=False)
        return velocity, residual, direction, next_rz
