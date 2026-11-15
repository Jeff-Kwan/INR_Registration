"""Image-independent neighbourhood directions for the shared cell solve."""
import torch


def normal_count(pairs, geometry):
    if geometry == 'independent':
        return pairs
    if geometry != 'orthogonal' or pairs not in (2, 3):
        raise ValueError('Orthogonal queries require two or three pairs')
    return 2


def query_directions(normals, pairs, geometry):
    """Preserve isotropic marginals; a three-frame reuses the two-frame RNG."""
    expected = normal_count(pairs, geometry)
    if normals.shape[-2:] != (expected, 3):
        raise ValueError('Unexpected neighbourhood normal shape')
    if geometry == 'independent':
        return normals / normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    first = normals[..., 0, :]
    first = first / first.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    second = normals[..., 1, :] - (normals[..., 1, :] * first).sum(-1, keepdim=True) * first
    second = second / second.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    if pairs == 2:
        return torch.stack((first, second), -2)
    return torch.stack((first, second, torch.linalg.cross(first, second, dim=-1)), -2)
