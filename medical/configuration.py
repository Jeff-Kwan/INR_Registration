"""Dataset-independent method configuration; old experiment keys are unsupported."""
import copy
import json
import math
from pathlib import Path

# One source of truth for the default recipe, including calls without a path.
DEFAULT = json.loads((Path(__file__).parent / 'configs/default.json').read_text())


def validate_config(config):
    if set(config)-{'working_grid','spatial_units'}!=set(DEFAULT): raise ValueError('Expected runtime, model, training, objective and recovery')
    for section in ('runtime','model','training','recovery'):
        optional = {'endpoint_sampling','tie_registration_points','structure_crop_sampling',
                    'optimizer','lr_schedule','objective_warmup_steps','objective_ramp_steps',
                    'first_layer_lr_factor','registration_weight_schedule','final_crop_fraction','temporal_sampling'} if section == 'training' else set()
        if section == 'model':
            optional = {'init_scale', 'first_omega_range', 'layer_init_scales', 'hidden_omega',
                        'balance_first_frequencies', 'first_omega_mode', 'output_init_scale'}
        if section == 'runtime':
            optional = {'training_execution', 'recovery_max_axis'}
        if set(config[section]) - optional != set(DEFAULT[section]): raise ValueError(f'Unexpected {section} keys')
    if config['runtime'].get('training_execution', 'standard') not in ('standard', 'captured'):
        raise ValueError('runtime.training_execution must be standard or captured')
    temporal=config['training'].get('temporal_sampling','independent')
    if temporal not in ('independent','sobol'):
        raise ValueError('training.temporal_sampling must be independent or sobol')
    if temporal=='sobol' and (config['runtime'].get('training_execution')!='captured'
                            or config['objective']['registration']['random_timesteps'] is not True):
        raise ValueError('Sobol temporal sampling requires captured execution and random_timesteps=true')
    if 'recovery_max_axis' in config['runtime']:
        cap = config['runtime']['recovery_max_axis']
        if type(cap) is not int or cap < 2:
            raise ValueError('runtime.recovery_max_axis must be an integer >= 2')
        if 'working_grid' in config or config.get('spatial_units', 'voxel') != 'voxel':
            raise ValueError('Capped recovery requires native voxel units without a working grid')
    if config.get('spatial_units','voxel') not in ('voxel','mm'):
        raise ValueError('spatial_units must be voxel or mm')
    if 'working_grid' in config and 'spatial_units' in config:
        raise ValueError('spatial_units is a native-grid option; working_grid already defines its units')
    if 'final_crop_fraction' in config['training']:
        start = config['objective']['registration'].get('crop_fraction')
        end = config['training']['final_crop_fraction']
        if any(type(v) not in (int, float) or not 0 < v <= 1 for v in (start, end)):
            raise ValueError('Linear crop schedule requires start and final fractions in (0, 1]')
        if config['training']['steps'] < 2 or 'crop_extent_mm' in config.get('working_grid', {}):
            raise ValueError('A linear fractional crop requires at least two steps and a fractional box')
    tied = config['training'].get('tie_registration_points', False)
    if type(tied) is not bool:
        raise ValueError('tie_registration_points must be boolean')
    if tied and ('endpoint_sampling' in config['training'] or
                 config['training']['batch_size'] != config['objective']['registration']['particles']):
        raise ValueError('Tied points require equal endpoint/registration counts and no structure sampler')
    if 'endpoint_sampling' in config['training']:
        from medical.registration.endpoint_sampling import validate_sampling
        validate_sampling(config['training']['endpoint_sampling'])
        if config['objective']['endpoint'] != 'mse':
            raise ValueError('Structure endpoint sampling currently requires MSE')
    if 'structure_crop_sampling' in config['training']:
        from medical.registration.endpoint_sampling import validate_sampling
        validate_sampling(config['training']['structure_crop_sampling'])
        if 'endpoint_sampling' in config['training']:
            raise ValueError('Structure crop sampling cannot combine with global endpoint sampling')
        reg = config['objective']['registration']
        if (reg.get('crop_fraction') is None
                and 'crop_extent_mm' not in config.get('working_grid', {})):
            raise ValueError('Structure crop sampling requires cropped training')
    if 'working_grid' in config:
        from medical.working_grid import validate_options
        validate_options(config['working_grid'])
    obj=config['objective']
    optional_objective = {'ncc','regularizer','endpoint_gw_grid_size','endpoint_lncc'}
    if set(obj) - optional_objective != set(DEFAULT['objective']) - optional_objective: raise ValueError('Unexpected objective keys')
    if obj['endpoint'] not in ('mse','ncc','gw_ncc','lncc'): raise ValueError('Endpoint must be mse, ncc, gw_ncc or lncc')
    if obj['endpoint'] == 'lncc':
        from medical.registration.endpoint_lncc import validate_windows
        validate_windows(obj.get('endpoint_lncc'))
        if ('working_grid' not in config or tied or 'endpoint_sampling' in config['training']):
            raise ValueError('Window LNCC requires a working grid and separate endpoint sampling')
    elif 'endpoint_lncc' in obj:
        raise ValueError('endpoint_lncc requires LNCC mode')
    if 'endpoint_gw_grid_size' in obj:
        size = obj['endpoint_gw_grid_size']
        if obj['endpoint'] != 'gw_ncc' or type(size) is not int or size < 2:
            raise ValueError('endpoint_gw_grid_size requires GW-NCC and an integer >= 2')
    if obj['endpoint'] in ('ncc','gw_ncc','lncc'):
        from medical.registration.ncc import ncc_options
        ncc_options(obj.get('ncc'))
    elif 'ncc' in obj: raise ValueError('MSE and endpoint NCC are separate modes')
    optional_registration = {'compile_decoder','sigma_mm','query_geometry'}
    if set(obj['registration']) - optional_registration != set(DEFAULT['objective']['registration']) - optional_registration: raise ValueError('Unexpected registration keys')
    from medical.registration.query_geometry import normal_count
    normal_count(obj['registration']['pairs_per_cell'], obj['registration'].get('query_geometry','independent'))
    if obj['registration']['registration_similarity'] not in ('ncc','gw_ncc'): raise ValueError('Unsupported registration similarity')
    if 'sigma_mm' in obj['registration']:
        sigma=obj['registration']['sigma_mm']
        if type(sigma) not in (int,float) or not math.isfinite(sigma) or sigma<=0:
            raise ValueError('registration.sigma_mm must be finite and positive')
    mode=obj.get('regularizer','pdlk')
    if mode not in ('pdlk','inverse_metric'): raise ValueError('Unsupported training regularizer')
    if mode=='inverse_metric':
        if (obj['endpoint']!='mse' or obj['weight_schedule']!='constant'
                or obj['c']!=4 or obj['eta']!=1e-6
                or obj['registration']['registration_similarity']!='ncc' or obj['registration']['pairs_per_cell']!=2):
            raise ValueError('Inverse metric requires MSE, c=4, eta=1e-6, two pairs and box NCC')
    for value in (obj['weight'],obj['registration']['weight'],obj['c']):
        if not math.isfinite(value) or value<0: raise ValueError('Loss weights and c must be nonnegative')
    if not math.isfinite(obj['eta']) or obj['eta']<=0: raise ValueError('eta must be positive')
    if obj['weight_schedule'] not in ('constant','linear','cosine'): raise ValueError('Invalid weight schedule')
    for name in ('compile_derivatives', 'compile_endpoints'):
        if type(config['runtime'][name]) is not bool: raise ValueError(f'runtime.{name} must be boolean')
    if type(obj['registration'].get('compile_decoder',False)) is not bool:
        raise ValueError('registration.compile_decoder must be boolean')
    for name in ('compile_velocity', 'recompute'):
        if type(obj['registration'][name]) is not bool: raise ValueError(f'registration.{name} must be boolean')
    model=config['model']
    output_scale = model.get('output_init_scale', 1.)
    if type(output_scale) not in (int, float) or not math.isfinite(output_scale) or output_scale <= 0:
        raise ValueError('Output initialization scale must be finite and positive')
    omega_mode = model.get('first_omega_mode', 'initialization')
    if (omega_mode not in ('initialization', 'parameterized')
            or (omega_mode == 'parameterized' and model.get('first_omega_range') is None)):
        raise ValueError('First omega mode must be initialization or parameterized with a range')
    balance = model.get('balance_first_frequencies', False)
    if type(balance) is not bool or (balance and model.get('first_omega_range') is None):
        raise ValueError('First-layer frequency balancing requires a boolean and an omega range')
    hidden_omega = model.get('hidden_omega', model['omega'])
    if type(hidden_omega) not in (int, float) or not math.isfinite(hidden_omega) or hidden_omega <= 0:
        raise ValueError('Hidden omega must be finite and positive')
    layer_scales = model.get('layer_init_scales')
    if layer_scales is not None:
        if (not isinstance(layer_scales, (list, tuple)) or len(layer_scales) != model['layers']
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       or value <= 0 for value in layer_scales)):
            raise ValueError('Layer initialization scales require one positive finite value per sine layer')
    first_omega_range = model.get('first_omega_range')
    if first_omega_range is not None:
        if (not isinstance(first_omega_range, (list, tuple)) or len(first_omega_range) != 2
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       or value <= 0 for value in first_omega_range)
                or first_omega_range[0] > first_omega_range[1]):
            raise ValueError('First-layer omega range must contain two ordered positive finite values')
    init_scale = model.get('init_scale', 1.)
    if type(init_scale) not in (int, float) or not math.isfinite(init_scale) or init_scale <= 0:
        raise ValueError('SIREN initialization scale must be finite and positive')
    if (any(type(model[key]) is not int or model[key]<1 for key in ('width','layers'))
            or type(model['omega']) not in (int,float) or not math.isfinite(model['omega']) or model['omega']<=0):
        raise ValueError('SIREN width/layers must be positive integers and omega finite and positive')
    from medical.registration.optimization import validate_optimization
    validate_optimization(config['training'])
    return config


def load_config(path=None):
    config=copy.deepcopy(DEFAULT) if path is None else json.loads(Path(path).read_text())
    return validate_config(config)
