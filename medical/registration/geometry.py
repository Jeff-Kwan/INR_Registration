"""Absolute sampler coordinates; vectors have reversed array-axis order."""
import torch
import torch.nn.functional as F

def identity_grid(shape, device):
    axes = torch.meshgrid(*(torch.arange(n, device=device, dtype=torch.float32)
                            for n in shape), indexing="ij")
    return torch.stack(tuple(reversed(axes)), -1)


def normalized_grid(mapping, shape, *, extent=None):
    if extent is None:
        extent = mapping.new_tensor([n - 1 for n in reversed(shape)])
    return 2 * mapping / extent - 1


def sample(volume, mapping, mode="bilinear", padding_mode="border", *, extent=None):
    """Sample a scalar [D,H,W] or vector [D,H,W,C] volume."""
    dimensions = mapping.shape[-1]
    vector = volume.ndim == dimensions + 1
    image = volume.movedim(-1, 0)[None] if vector else volume[None, None]
    result = F.grid_sample(image, normalized_grid(mapping, volume.shape[:dimensions], extent=extent)[None],
                           mode=mode, padding_mode=padding_mode, align_corners=True)[0]
    return result.movedim(0, -1) if vector else result[0]
