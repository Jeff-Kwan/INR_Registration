"""Independent higher-order autograd reference for tests and benchmarks."""
import torch
from torch.func import functional_call, jvp


def autograd_reference(source, target, coordinates, time, scale):
    """Differentiate the actual intensity, not the manual derivative function."""
    x = coordinates.detach().requires_grad_(True)
    first, second = dict(source.named_parameters()), dict(target.named_parameters())
    def image(t):
        parameters = {k: torch.lerp(v, second[k], t) for k, v in first.items()}
        return functional_call(source, parameters, (x.reshape(-1, x.shape[-1]),)).reshape(x.shape[:-1])
    intensity, q = jvp(image, (time,), (torch.ones_like(time),))
    scale = x.new_tensor(scale)
    g, = torch.autograd.grad(intensity.sum(), x, create_graph=True)
    g = g * scale
    rows = [torch.autograd.grad(g[..., i].sum(), x, create_graph=True)[0] * scale
            for i in range(x.shape[-1])]
    mixed, = torch.autograd.grad(q.sum(), x, create_graph=True)
    return g, q, torch.stack(rows, -2), mixed * scale

