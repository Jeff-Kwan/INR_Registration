"""Learn2Reg OASIS fits of single-field SIRENs with explicit time conditioning."""
from functools import partial

from medical import sirenpath_ablation_method
from medical.models.conditioned_siren import ConditionedSiren, conditioned_affines, time_derivative
from medical.registration.captured_objective import transport_registration
from medical.registration.captured_training import fit_field
from medical.registration.derivatives import _layer_curvature_chain
from medical.registration.objective import _endpoint_kernel
from medical.registration.training_integrators import DEFAULT_INTEGRATOR


def joint_loss(parameters, conditioning, omegas, source_coordinates, target_coordinates,
               source_truth, target_truth, query_coordinates, spatial_scale, positions, moving,
               fixed, origin, domain, extent, times, sigma, grid_size, c, eta, local_transport_weight,
               registration_weight, ncc_options, registration_options=None,
               integrator=DEFAULT_INTEGRATOR, parameter_l2=0., endpoint_weight=1.,
               brightness_weight=1., endpoint_mode='ncc', transport_enabled=True,
               registration_enabled=True):
    direction = time_derivative(parameters, conditioning)
    source = conditioned_affines(parameters, direction, 0.)
    target = conditioned_affines(parameters, direction, 1.)
    fuse_affines = source[0][0].is_cuda and source[0][0].shape == (512, 3)
    terms = _endpoint_kernel(source, target, omegas, source_coordinates, target_coordinates,
                             source_truth, target_truth, endpoint_mode, ncc_options,
                             fuse_scalar_affines=fuse_affines, parameter_l2=parameter_l2,
                             endpoint_weight=endpoint_weight)
    samples = (tuple(conditioned_affines(parameters, direction, time) for time in times)
               if transport_enabled or registration_enabled else ())
    return transport_registration(terms, samples, direction, omegas, query_coordinates, spatial_scale,
                                  positions, moving, fixed, origin, domain, extent, sigma, grid_size, c,
                                  eta, local_transport_weight, registration_weight, registration_options, integrator,
                                  brightness_weight, _layer_curvature_chain, registration_enabled)


def fit_conditioned(moving, fixed, config, *, architecture, report=None, affine=None,
                    checkpoint_callback=None):
    if config['objective']['parameter_l2'] != 0:
        raise ValueError('A time-conditioned field requires zero endpoint parameter penalty')
    return fit_field(moving, fixed, config, lambda: ConditionedSiren(architecture, **config['model']),
                     ConditionedSiren.training_arguments, joint_loss, report=report, affine=affine,
                     checkpoint_callback=checkpoint_callback)


def run_case(case, config, device, *, checkpoint_callback=None):
    fields = dict(config)
    architecture = fields.pop('architecture')
    fit = partial(fit_conditioned, architecture=architecture)
    result = sirenpath_ablation_method.run_case(case, fields, device,
                                                checkpoint_callback=checkpoint_callback, field_fitter=fit)
    result['metrics'].update(method={'xyzt': 'XYZT input', 'film': 'Hidden-layer FiLM'}[architecture],
                             control=architecture)
    result['checkpoint']['config'] = config
    return result
