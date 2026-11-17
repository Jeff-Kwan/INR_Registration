"""Two orthogonal neighbourhood directions for each shared cell solve."""
import torch


def normal_count(pairs, geometry):
    if geometry != 'orthogonal' or pairs != 2:
        raise ValueError('Shared-grid queries require two orthogonal pairs')
    return 2


def query_directions(normals, pairs, geometry):
    if normals.shape[-2:] != (normal_count(pairs, geometry), 3):
        raise ValueError('Unexpected neighbourhood normal shape')
    first = normals[..., 0, :]
    first = first / first.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    second = normals[..., 1, :] - (normals[..., 1, :] * first).sum(-1, keepdim=True) * first
    second = second / second.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.stack((first, second), -2)
