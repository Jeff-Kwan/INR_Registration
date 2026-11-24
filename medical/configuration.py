"""The canonical paired-SIREN recipe and adjustable component weights."""
import copy
import json
import math
from pathlib import Path

DEFAULT = json.loads((Path(__file__).parent / 'configs/default.json').read_text())


def _number(value, name, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')


def validate_config(config, *, allow_ablation_mse=False):
    """Validate recipe settings, including independently adjustable loss weights."""
    if not isinstance(config, dict) or set(config) != set(DEFAULT):
        raise ValueError('Expected only runtime, model, training, objective and recovery')
    for section in DEFAULT:
        if not isinstance(config[section], dict) or set(config[section]) != set(DEFAULT[section]):
            raise ValueError(f'Unexpected {section} keys')
    runtime, model, training, objective, recovery = (config[key] for key in DEFAULT)
    registration = objective['registration']
    if not isinstance(registration, dict) or set(registration) != set(DEFAULT['objective']['registration']):
        raise ValueError('Unexpected registration keys')
    if runtime['training_execution'] != 'captured':
        raise ValueError('Training uses the captured CUDA FP32 execution path')
    for name in ('compile_derivatives', 'compile_endpoints'):
        if runtime[name] is not True:
            raise ValueError(f'runtime.{name} must be true for captured training')
    for name in ('compile_decoder', 'compile_velocity', 'random_timesteps'):
        if registration[name] is not True:
            raise ValueError(f'registration.{name} must be true')
    if registration['recompute'] is not False:
        raise ValueError('Captured training does not use activation recomputation')
    for value, name, minimum in [
            (runtime['seed'], 'runtime.seed', 0),
            (runtime['inference_chunk'], 'runtime.inference_chunk', 1),
            (runtime['recovery_max_axis'], 'runtime.recovery_max_axis', 2),
            (model['width'], 'model.width', 1), (model['layers'], 'model.layers', 1),
            (training['steps'], 'training.steps', 1),
            (training['batch_size'], 'training.batch_size', 1),
            (training['log_every'], 'training.log_every', 1),
            (registration['grid_size'], 'registration.grid_size', 2),
            (registration['particles'], 'registration.particles', 1),
            (registration['chunk'], 'registration.chunk', 4),
            (recovery['intervals'], 'recovery.intervals', 1),
            (recovery['iterations'], 'recovery.iterations', 1),
            (recovery['substeps'], 'recovery.substeps', 1)]:
        if type(value) is not int or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if not isinstance(runtime['device'], str) or not runtime['device'].startswith('cuda'):
        raise ValueError('The canonical training recipe requires a CUDA device')
    if training['integrator'] != 'rk2' or training['temporal_sampling'] != 'sobol':
        raise ValueError('Training uses two RK2 half-steps with scrambled Sobol times')
    allowed_endpoints = ('ncc', 'mse') if allow_ablation_mse else ('ncc',)
    if objective['endpoint'] not in allowed_endpoints:
        raise ValueError('Endpoint NCC is canonical; MSE requires the explicit ablation entry point')
    schedule = objective['weight_schedule']
    if schedule != 'constant':
        if (not isinstance(schedule, dict) or set(schedule) != {'name', 'start_weight', 'decay_steps'}
                or schedule['name'] != 'cosine'):
            raise ValueError('Local transport weight schedule must be constant or an explicit cosine schedule')
        _number(schedule['start_weight'], 'Local transport start weight')
        if type(schedule['decay_steps']) is not int or not 1 <= schedule['decay_steps'] <= training['steps']:
            raise ValueError('Invalid local transport decay_steps')
    if objective['endpoint'] == 'mse' and objective['ncc']['moment_weight'] != 0:
        raise ValueError('MSE ablations disable the NCC-specific moment penalty')
    if registration['registration_similarity'] != 'gw_ncc':
        raise ValueError('Registration uses the global/window GW-NCC mixture')
    if registration['query_geometry'] != 'orthogonal' or registration['pairs_per_cell'] != 2:
        raise ValueError('Each shared-grid cell uses two orthogonal query pairs')
    if registration['gaussian_kernel'] != 'three_tap' or registration['gw_ncc_kernel'] != 'three_tap':
        raise ValueError('Velocity and NCC moments use three-tap Gaussian filtering')
    for section, names in [(model, ('omega', 'output_init_scale')),
                           (training, ('lr', 'lr_final_factor')),
                           (objective, ('eta',)),
                           (registration, ('sigma_cells', 'gw_ncc_sigma_cells')),
                           (recovery, ('ridge', 'tolerance'))]:
        for name in names:
            _number(section[name], name, positive=True)
    for name in ('weight', 'endpoint_weight', 'brightness_weight', 'parameter_l2', 'c'):
        _number(objective[name], f'objective.{name}')
    _number(registration['weight'], 'registration.weight')
    _number(training['grad_clip'], 'training.grad_clip')
    _number(recovery['smoothness'], 'recovery.smoothness')
    for name, maximum in [('crop_fraction', 1.), ('particle_sampling_fraction', 1.),
                          ('pair_offset_fraction', .5)]:
        value = registration[name]
        _number(value, f'registration.{name}', positive=True)
        if value > maximum:
            raise ValueError(f'registration.{name} must be <= {maximum}')
    from medical.registration.ncc import ncc_options
    ncc_options(objective['ncc'])
    from medical.registration.optimization import validate_optimization
    validate_optimization(training)
    return config


def load_config(path=None):
    config = copy.deepcopy(DEFAULT) if path is None else json.loads(Path(path).read_text())
    return validate_config(config)
