"""Opt-in optimizer and learning-rate controls for convergence experiments."""
import math

import torch


def validate_optimization(settings):
    registration_schedule = settings.get('registration_weight_schedule')
    if registration_schedule is not None:
        if (not isinstance(registration_schedule, dict)
                or set(registration_schedule) != {'name', 'start_weight', 'decay_steps'}
                or registration_schedule['name'] not in ('linear', 'cosine')):
            raise ValueError('Registration schedule requires name, start_weight and decay_steps')
        start_weight = registration_schedule['start_weight']
        decay_steps = registration_schedule['decay_steps']
        if (type(start_weight) not in (int, float) or not math.isfinite(start_weight)
                or start_weight < 0):
            raise ValueError('Registration start weight must be finite and nonnegative')
        if type(decay_steps) is not int or not 2 <= decay_steps <= settings['steps']:
            raise ValueError('Registration decay_steps must be between 2 and training steps')
    first_lr = settings.get('first_layer_lr_factor', 1.)
    if type(first_lr) not in (int, float) or not math.isfinite(first_lr) or first_lr <= 0:
        raise ValueError('First-layer learning-rate factor must be finite and positive')
    options = settings.get('optimizer', {})
    if not isinstance(options, dict) or set(options) - {
            'name', 'betas', 'weight_decay', 'momentum', 'muon_lr_multiplier'}:
        raise ValueError('Unsupported optimizer settings')
    name = options.get('name', 'adam')
    if name not in ('adam', 'adamw', 'radam', 'sgd', 'muon'):
        raise ValueError('Unsupported optimizer')
    betas = options.get('betas', [.9, .999])
    if (not isinstance(betas, (list, tuple)) or len(betas) != 2
            or any(type(value) not in (int, float) or not 0 <= value < 1 for value in betas)):
        raise ValueError('Optimizer betas must contain two values in [0, 1)')
    decay = options.get('weight_decay', 0.)
    if type(decay) not in (int, float) or not math.isfinite(decay) or decay < 0:
        raise ValueError('Weight decay must be finite and nonnegative')
    momentum = options.get('momentum', .95 if name == 'muon' else .9)
    if type(momentum) not in (int, float) or not 0 < momentum < 1:
        raise ValueError('Momentum must be in (0, 1)')
    multiplier = options.get('muon_lr_multiplier', 1.)
    if type(multiplier) not in (int, float) or not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('Muon learning-rate multiplier must be finite and positive')
    if name != 'muon' and 'muon_lr_multiplier' in options:
        raise ValueError('muon_lr_multiplier requires Muon')
    if name not in ('muon', 'sgd') and 'momentum' in options:
        raise ValueError('Explicit momentum requires SGD or Muon')
    if name == 'sgd' and 'betas' in options:
        raise ValueError('SGD does not use betas')
    schedule = settings.get('lr_schedule', {})
    if not isinstance(schedule, dict) or set(schedule) - {'name', 'warmup_fraction', 'start_factor'}:
        raise ValueError('Unsupported learning-rate schedule settings')
    schedule_name = schedule.get('name', 'exponential')
    if schedule_name not in ('exponential', 'constant', 'cosine', 'onecycle'):
        raise ValueError('Unsupported learning-rate schedule')
    warmup = schedule.get('warmup_fraction', .2 if schedule_name == 'onecycle' else 0.)
    if type(warmup) not in (int, float) or not 0 <= warmup < 1:
        raise ValueError('Learning-rate warmup fraction must be in [0, 1)')
    if schedule_name == 'onecycle' and warmup == 0:
        raise ValueError('OneCycle requires a positive warmup fraction')
    start = schedule.get('start_factor', .1)
    if type(start) not in (int, float) or not 0 < start <= 1:
        raise ValueError('Learning-rate start factor must be in (0, 1]')
    if schedule_name in ('constant', 'exponential') and set(schedule) - {'name'}:
        raise ValueError('Warmup options require cosine or OneCycle')
    for key in ('objective_warmup_steps', 'objective_ramp_steps'):
        value = settings.get(key, 0)
        if type(value) is not int or not 0 <= value < settings['steps']:
            raise ValueError(f'{key} must be an integer in [0, steps)')
    if settings.get('objective_warmup_steps', 0) + settings.get('objective_ramp_steps', 0) >= settings['steps']:
        raise ValueError('Objective warmup and ramp must finish before the final step')


def make_optimizers(pair, settings):
    options = settings.get('optimizer', {})
    name = options.get('name', 'adam')
    lr = settings['lr']
    decay = options.get('weight_decay', 0.)
    betas = tuple(options.get('betas', [.9, .999]))
    first_parameters = [value for model in (pair.source, pair.target)
                        for value in model.net[0].parameters()]
    first_ids = {id(value) for value in first_parameters}

    def groups(parameters):
        parameters = list(parameters)
        factor = settings.get('first_layer_lr_factor', 1.)
        if factor == 1:
            return parameters
        first = [value for value in parameters if id(value) in first_ids]
        other = [value for value in parameters if id(value) not in first_ids]
        return [{'params': first, 'lr': lr * factor}, {'params': other, 'lr': lr}]

    if name == 'sgd':
        optimizers = [torch.optim.SGD(groups(pair.parameters()), lr=lr,
                                     momentum=options.get('momentum', .9),
                                     nesterov=True, weight_decay=decay)]
    elif name == 'muon':
        matrices = [layer.linear.weight for model in (pair.source, pair.target)
                    for layer in model.net[1:-1]]
        matrix_ids = {id(value) for value in matrices}
        remaining = [value for value in pair.parameters() if id(value) not in matrix_ids]
        optimizers = [torch.optim.AdamW(groups(remaining), lr=lr, betas=betas, weight_decay=decay)]
        if matrices:
            optimizers.append(torch.optim.Muon(
                matrices, lr=lr * options.get('muon_lr_multiplier', 1.),
                momentum=options.get('momentum', .95), weight_decay=decay,
                adjust_lr_fn='match_rms_adamw'))
    else:
        constructor = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW,
                       'radam': torch.optim.RAdam}[name]
        optimizers = [constructor(groups(pair.parameters()), lr=lr, betas=betas, weight_decay=decay)]
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group['convergence_base_lr'] = group['lr']
    return optimizers


def learning_rate_factor(settings, step):
    schedule = settings.get('lr_schedule', {})
    name = schedule.get('name', 'exponential')
    if name == 'exponential':
        # Preserve the existing default, including its final-step convention.
        return settings['lr_final_factor'] ** ((step - 1) / settings['steps'])
    if name == 'constant':
        return 1.
    fraction = (step - 1) / max(settings['steps'] - 1, 1)
    warmup = schedule.get('warmup_fraction', .2 if name == 'onecycle' else 0.)
    if warmup and fraction < warmup:
        progress = fraction / warmup
        rise = .5 * (1 - math.cos(math.pi * progress)) if name == 'onecycle' else progress
        return schedule.get('start_factor', .1) + (1 - schedule.get('start_factor', .1)) * rise
    progress = (fraction - warmup) / (1 - warmup)
    final = settings['lr_final_factor']
    return final + (1 - final) * .5 * (1 + math.cos(math.pi * progress))


def objective_factor(settings, step):
    warmup = settings.get('objective_warmup_steps', 0)
    if step <= warmup:
        return 0.
    ramp = settings.get('objective_ramp_steps', 0)
    return min(1., (step - warmup) / ramp) if ramp else 1.


def registration_weight(settings, final_weight, step):
    schedule = settings.get('registration_weight_schedule')
    if schedule is None:
        return final_weight
    progress = min(1., (step - 1) / (schedule['decay_steps'] - 1))
    remaining = .5 * (1 + math.cos(math.pi * progress)) if schedule['name'] == 'cosine' else 1 - progress
    return final_weight + (schedule['start_weight'] - final_weight) * remaining
