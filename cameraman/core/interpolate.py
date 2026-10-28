"""Decode a parameter path and the pixelwise cross-fade control."""
import torch
from cameraman.core.inr import decode_parameter_sequence


@torch.no_grad()
def interpolate_weights(model, source_state, target_state, coordinates, size, frames):
    times = torch.linspace(0, 1, frames, device=coordinates.device, dtype=coordinates.dtype)
    return decode_parameter_sequence(model, source_state, target_state, coordinates, times).reshape(frames, size, size)


def crossfade(source, target, frames):
    t = torch.linspace(0, 1, frames, device=source.device)[:, None, None]
    return (1 - t) * source + t * target
