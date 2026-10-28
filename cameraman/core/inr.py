"""Paired 2D SIRENs with a pointwise standard/relaxed PD-LK objective."""
import copy
import gc
import math

import torch
from torch import nn
from tqdm.auto import tqdm

from cameraman.core.affine_pdlk import affine_temporal_loss
from cameraman.core.lk_loss import regularization_weight


def initialize_endpoint_pair(factory,device,mode='shared',target_seed=43):
    """Keep source initialization and subsequent sampling RNG matched across modes."""
    if mode not in ('shared','independent'):
        raise ValueError('Initialization must be shared or independent')
    device=torch.device(device)
    initial=factory().to(device)
    source=copy.deepcopy(initial)
    if mode=='shared':
        target=copy.deepcopy(initial)
    else:
        devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(target_seed)
            target=factory().to(device)
    return initial,source,target


class SineLayer(nn.Module):
    def __init__(self, inputs, outputs, omega=30., first=False):
        super().__init__()
        self.linear = nn.Linear(inputs, outputs)
        self.omega = omega
        self.activation = torch.sin
        bound = 1 / inputs if first else math.sqrt(6 / inputs) / omega
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        values = self.linear(x)
        if not torch.is_grad_enabled():
            values.mul_(self.omega)
            return values.sin_() if self.activation is torch.sin else self.activation(values)
        return self.activation(self.omega * values)


class Siren(nn.Module):
    def __init__(self, width=128, layers=3, omega=30.):
        super().__init__()
        blocks = [SineLayer(2, width, omega, first=True)]
        blocks += [SineLayer(width, width, omega) for _ in range(layers - 1)]
        output = nn.Linear(width, 1)
        bound = math.sqrt(6 / width) / omega
        with torch.no_grad():
            output.weight.uniform_(-bound, bound)
            output.bias.zero_()
        self.net = nn.Sequential(*blocks, output)

    def forward(self, coordinates):
        return self.net(coordinates)


def prepare_evaluation(*models):
    """Release training-only buffers before full trajectory decoding."""
    for model in models:
        model.zero_grad(set_to_none=True)
    gc.collect()
    devices = {parameter.device for model in models for parameter in model.parameters()
               if parameter.is_cuda}
    for device in devices:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def decode_parameter_sequence(model, source_parameters, target_parameters, coordinates, times):
    """Batch time-dependent weights with vmap; each network sees the entire image.

    Returns [T, H*W, 1]. No spatial splits, time loops, detaches, or activation
    checkpoint recomputation. The parameter construction preserves both endpoints.
    """
    parameters = {
        name: torch.lerp(value.unsqueeze(0), target_parameters[name].unsqueeze(0),
                         times.to(value.dtype).reshape((-1,) + (1,) * value.ndim))
        for name, value in source_parameters.items()
    }

    def forward(state):
        return torch.func.functional_call(model, state, (coordinates,))

    return torch.vmap(forward)(parameters)

def validate_training_settings(settings):
    """Validate the supported fitter options without creating an optimizer."""
    for key in ('steps', 'batch_size', 'log_every', 'lk_points'):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('lr', 'lk_ridge', 'lk_weight', 'lk_gradient_weight', 'lk_affine_rho', 'grad_clip', 'lr_final_factor'):
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if settings['lr'] <= 0 or settings['lk_ridge'] <= 0:
        raise ValueError('lr and lk_ridge must be positive')
    if not 0 < settings['lr_final_factor'] <= 1:
        raise ValueError('lr_final_factor must lie in (0,1]')
    if settings['lr_schedule'] not in ('linear', 'exponential'):
        raise ValueError('lr_schedule must be linear or exponential')
    if settings['lk_weight_schedule'] not in ('constant', 'linear', 'cosine'):
        raise ValueError('lk_weight_schedule must be constant, linear or cosine')
    if type(settings['lk_differentiate_solve']) is not bool:
        raise ValueError('lk_differentiate_solve must be boolean')


def pair_objective(source, target, coordinates, source_values, target_values,
                   points, time_point, shape, *, lk_weight=.1, lk_ridge=1e-6,
                   lk_gradient_weight=4., lk_affine_rho=.4, lk_differentiate_solve=False):
    """Endpoint MSE plus minimized pointwise PD-LK; contains no optimizer.

    points are independent normalized queries [N,1,2] at one shared time.
    Passing points=None disables the regularizer and requires lk_weight=0.
    """
    source_mse = (source(coordinates) - source_values).square().mean()
    target_mse = (target(coordinates) - target_values).square().mean()
    if points is None:
        if lk_weight != 0:
            raise ValueError('Nonzero lk_weight requires regularization points')
        lk_loss = source_mse.new_zeros(())
    else:
        lk_loss = affine_temporal_loss(
            source, target, points, time_point, shape, ridge=lk_ridge,
            c=lk_gradient_weight, rho=lk_affine_rho,
            differentiate_solve=lk_differentiate_solve)
    weighted_lk = lk_weight * lk_loss
    return {'source_batch_mse': source_mse, 'target_batch_mse': target_mse,
            'lk_loss': lk_loss, 'weighted_lk': weighted_lk,
            'total_loss': source_mse + target_mse + weighted_lk}


def fit_pair(source_model, target_model, coordinates, source_image, target_image,
             steps, batch_size, lr, log_every=10, lr_final_factor=1.,
             lr_schedule='exponential', lk_weight=.1, lk_points=1960,
             lk_ridge=1e-6, lk_gradient_weight=4., lk_affine_rho=.4,
             lk_differentiate_solve=False, lk_weight_schedule='constant', grad_clip=0.):
    """Fit paired SIRENs with matched endpoint samples and pointwise PD-LK.

    Adam, native-pixel derivatives, a shared uniform time, and independent
    full-domain spatial queries follow the same convention as the 3D pipeline.
    rho=0 selects exact standard PD-LK; positive rho selects its relaxation.
    """
    validate_training_settings(locals())
    if type(source_model) is not Siren or type(target_model) is not Siren:
        raise ValueError('Paired SIRENs are required')
    if source_image.ndim != 2 or source_image.shape != target_image.shape or min(source_image.shape) < 2:
        raise ValueError('Endpoint images must share a 2D grid with axes >=2')
    if coordinates.shape != (source_image.numel(), 2):
        raise ValueError('Expected one normalized xy coordinate per endpoint pixel')
    parameters = list(source_model.parameters()) + list(target_model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr, fused=source_image.is_cuda)
    source_pixels, target_pixels = source_image.reshape(-1, 1), target_image.reshape(-1, 1)
    history = []
    source_model.train()
    target_model.train()
    with tqdm(total=steps, desc='Training', dynamic_ncols=True, mininterval=0, miniters=1) as progress:
        for step in range(1, steps + 1):
            if lr_schedule == 'linear':
                fraction = (step - 1) / max(steps - 1, 1)
                current_lr = lr * (1 + (lr_final_factor - 1) * fraction)
            else:
                current_lr = lr * lr_final_factor ** ((step - 1) / steps)
            for group in optimizer.param_groups:
                group['lr'] = current_lr
            if batch_size < source_pixels.shape[0]:
                indices = torch.randint(source_pixels.shape[0], (batch_size,), device=source_image.device)
                xy, first, second = coordinates[indices], source_pixels[indices], target_pixels[indices]
            else:
                xy, first, second = coordinates, source_pixels, target_pixels
            optimizer.zero_grad(set_to_none=True)
            points, time_point = None, None
            if lk_weight > 0:
                time_point = torch.rand(1, device=coordinates.device, dtype=coordinates.dtype)[0]
                points = torch.rand(lk_points, 1, 2, device=coordinates.device, dtype=coordinates.dtype) * 2 - 1
            weight = regularization_weight({'weight': lk_weight, 'weight_schedule': lk_weight_schedule}, step, steps)
            values = pair_objective(
                source_model, target_model, xy, first, second, points, time_point,
                source_image.shape, lk_weight=weight, lk_ridge=lk_ridge,
                lk_gradient_weight=lk_gradient_weight, lk_affine_rho=lk_affine_rho,
                lk_differentiate_solve=lk_differentiate_solve)
            values['total_loss'].backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
            optimizer.step()
            if step % log_every == 0 or step == steps:
                samples = [value.detach() for value in values.values()]
                if time_point is not None:
                    samples.append(time_point)
                scalars = torch.stack(samples).cpu().tolist()
                record = {'step': step, **dict(zip(values, scalars)),
                          'time': scalars[-1] if time_point is not None else None,
                          'learning_rate': current_lr, 'regularization_weight': weight}
                history.append(record)
                progress.set_postfix(src=f"{record['source_batch_mse']:.4g}",
                                     tgt=f"{record['target_batch_mse']:.4g}",
                                     lk=f"{record['lk_loss']:.4g}", refresh=False)
                progress.update(step - progress.n)
    source_model.eval()
    target_model.eval()
    return history
