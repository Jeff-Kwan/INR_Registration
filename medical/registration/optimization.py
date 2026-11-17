"""AdamW with canonical cosine LR or the explicitly requested OneCycle tuning option."""
import math


def validate_optimization(settings):
    options = settings['optimizer']
    if not isinstance(options, dict) or set(options) != {'name', 'weight_decay'} or options['name'] != 'adamw':
        raise ValueError('The canonical optimizer is AdamW with weight_decay')
    decay = options['weight_decay']
    if type(decay) not in (int, float) or not math.isfinite(decay) or decay < 0:
        raise ValueError('Weight decay must be finite and nonnegative')
    schedule = settings['lr_schedule']
    if not isinstance(schedule, dict):
        raise ValueError('Expected a learning-rate schedule dictionary')
    if schedule.get('name') == 'cosine':
        if schedule != {'name': 'cosine'}:
            raise ValueError('Cosine LR has no warmup options')
    elif schedule.get('name') == 'onecycle':
        if set(schedule) != {'name', 'warmup_fraction', 'start_factor'}:
            raise ValueError('OneCycle requires warmup_fraction and start_factor')
        warmup, start = schedule['warmup_fraction'], schedule['start_factor']
        if type(warmup) not in (int, float) or not math.isfinite(warmup) or not 0 < warmup < 1:
            raise ValueError('OneCycle warmup_fraction must be in (0, 1)')
        if type(start) not in (int, float) or not math.isfinite(start) or not 0 < start <= 1:
            raise ValueError('OneCycle start_factor must be in (0, 1]')
        if settings['steps'] < 2:
            raise ValueError('OneCycle requires at least two updates')
    else:
        raise ValueError('Learning-rate schedule must be cosine or onecycle')
    schedule = settings['registration_weight_schedule']
    if (not isinstance(schedule, dict) or set(schedule) != {'name', 'start_weight', 'decay_steps'}
            or schedule['name'] != 'cosine'):
        raise ValueError('The registration coefficient schedule is cosine')
    start, steps = schedule['start_weight'], schedule['decay_steps']
    if type(start) not in (int, float) or not math.isfinite(start) or start < 0:
        raise ValueError('Registration start weight must be finite and nonnegative')
    if type(steps) is not int or not 1 <= steps <= settings['steps']:
        raise ValueError('Registration decay_steps must be between 1 and training steps')




def learning_rate_factor(settings, step):
    progress = min(1., max(0., (step - 1) / max(settings['steps'] - 1, 1)))
    schedule = settings['lr_schedule']
    if schedule['name'] == 'onecycle':
        # Historical LR-only recipe: keep AdamW betas fixed. Host-side scalar
        # evaluation feeds the existing live CUDA-graph LR tensor each update.
        warmup = schedule['warmup_fraction']
        if progress < warmup:
            rise_progress = progress / warmup
            rise = .5 * (1 - math.cos(math.pi * rise_progress))
            start = schedule['start_factor']
            return start + (1 - start) * rise
        progress = (progress - warmup) / (1 - warmup)
    final = settings['lr_final_factor']
    return final + (1 - final) * .5 * (1 + math.cos(math.pi * progress))


def registration_weight(settings, final_weight, step):
    # A zero coefficient disables the entire component, including its schedule.
    if final_weight == 0:
        return 0.
    schedule = settings['registration_weight_schedule']
    progress = min(1., (step - 1) / max(schedule['decay_steps'] - 1, 1))
    remaining = .5 * (1 + math.cos(math.pi * progress))
    return final_weight + (schedule['start_weight'] - final_weight) * remaining


def pdlk_weight(objective, step):
    """A zero PD-LK coefficient disables its loss throughout the schedule."""
    final=objective['weight']
    if final==0 or objective['weight_schedule']=='constant':return final
    schedule=objective['weight_schedule']
    progress=min(1.,max(0.,(step-1)/max(schedule['decay_steps']-1,1)))
    return final+(schedule['start_weight']-final)*.5*(1+math.cos(math.pi*progress))
