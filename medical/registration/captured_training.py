"""SIRENPath training with endpoint NCC, local transport loss and trajectory-based registration."""
import json
import time
import torch

from medical.models.siren import PairedSiren
from medical.models.siren_functional import siren_parameters
from medical.registration.symmetric_registration import SymmetricPairRegistration
from medical.registration.optimization import learning_rate_factor, registration_weight, local_transport_weight as scheduled_local_transport_weight
from medical.registration.ncc import ncc_options
from medical.registration.captured_objective import joint_loss
from medical.registration.temporal_graph_sampling import SobolTimeInputs
from medical.registration.training_integrators import integration_metadata

_LAST_COMPILE_SIGNATURE = None


def fit_pair(moving, fixed, config, *, report=None, affine=None,
             checkpoint_callback=None, allow_ablation_mse=False):
    return fit_field(moving, fixed, config,
                     lambda: PairedSiren(3, **config['model']), _paired_arguments, joint_loss,
                     report=report, affine=affine,
                     checkpoint_callback=checkpoint_callback, allow_ablation_mse=allow_ablation_mse)


def _paired_arguments(pair):
    source, omegas = siren_parameters(pair.source)
    target, target_omegas = siren_parameters(pair.target)
    if omegas != target_omegas:
        raise ValueError('Endpoint frequencies must match')
    return source, target, omegas


def fit_field(moving, fixed, config, create_field, field_arguments, loss_function, *,
              report=None, affine=None,
              checkpoint_callback=None, allow_ablation_mse=False):
    """Shared captured update for differentiable fields with the same loss inputs."""
    from medical.configuration import validate_config
    validate_config(config, allow_ablation_mse=allow_ablation_mse)
    settings, objective = config['training'], config['objective']
    if (not moving.is_cuda or moving.dtype != torch.float32 or fixed.dtype != moving.dtype
            or fixed.device != moving.device):
        raise ValueError('Canonical captured training requires CUDA FP32 endpoint images')
    torch.manual_seed(config['runtime']['seed'])
    torch.set_float32_matmul_precision('highest')
    pair = create_field().to(moving.device)
    pair.training_metadata = {
        'transport_objective': 'local_transport_loss', 'local_transport_weight': objective['weight'],
        'local_transport_weight_schedule': objective['weight_schedule'],
        'endpoint_objective': objective['endpoint'], 'endpoint_weight': objective['endpoint_weight'],
        'endpoint_sampling': 'shared_registration_crop', 'spatial_units': 'voxel',
        **integration_metadata('rk2', True),
        'execution': 'captured',
        'matmul_precision': torch.get_float32_matmul_precision(),
        'sampling': 'native spatial CUDA RNG; scrambled2D Sobol temporal quadrature',
        'training_time_sampling': 'scrambled_sobol_per_half',
        'temporal_sampling': 'sobol', 'query_geometry': 'orthogonal',
        'registration_similarity': 'gw_ncc',
        'registration_weight_schedule': settings['registration_weight_schedule'],
        'parameter_l2': objective['parameter_l2'],
        'brightness_weight': objective['brightness_weight'], 'gradient_constancy_weight': objective['c'],
        'optimization': {'optimizer': settings['optimizer'], 'lr_schedule': settings['lr_schedule']}}
    branch = SymmetricPairRegistration(
        moving, fixed, affine, seed=config['runtime']['seed'], c=objective['c'], eta=objective['eta'],
        compile_derivatives=True, training_integrator='rk2', brightness_weight=objective['brightness_weight'],
        **{k: v for k, v in objective['registration'].items() if k != 'weight'})
    rate = torch.tensor(settings['lr'], device=moving.device)
    optimizer = torch.optim.AdamW(pair.parameters(), lr=rate,
                                  weight_decay=settings['optimizer']['weight_decay'],
                                  fused=True, capturable=True)
    inputs = SobolTimeInputs(branch, settings['batch_size'], config['runtime']['seed'], settings['steps'])
    parameters = field_arguments(pair)
    local_transport_weight = torch.zeros((), device=moving.device)
    registration_coefficient = torch.zeros((), device=moving.device)
    # Reuse compiled code for equal geometries, with fresh parameters and optimizer state.
    global _LAST_COMPILE_SIGNATURE
    signature = (loss_function, tuple(moving.shape), str(moving.dtype), str(moving.device),
                 json.dumps(config, sort_keys=True), json.dumps(pair.model_config, sort_keys=True))
    if signature != _LAST_COMPILE_SIGNATURE:
        torch._dynamo.reset_code(loss_function.__code__)
        _LAST_COMPILE_SIGNATURE = signature
    compiled = torch.compile(loss_function, fullgraph=True, dynamic=False, options={'triton.cudagraphs': False})
    options_ncc = ncc_options(objective['ncc'])

    def update():
        data = inputs.draw()
        pair.zero_grad(set_to_none=False)
        terms, finite = compiled(
            *parameters, data['coordinates'], data['coordinates'],
            data['source_truth'], data['target_truth'], data['queries'], branch.spatial_scale,
            (data['forward_positions'], data['reverse_positions']), moving, fixed, data['origin'],
            branch.domain_extent, branch.extent, data['times'], tuple(branch.sigma), branch.grid_size,
            branch.c, branch.eta, local_transport_weight, registration_coefficient, options_ncc,
            objective['registration'], 'rk2', objective['parameter_l2'], objective['endpoint_weight'], objective['brightness_weight'],
            endpoint_mode=objective['endpoint'], transport_enabled=objective['weight'] > 0,
            registration_enabled=objective['registration']['weight'] > 0)
        torch._assert_async(finite & torch.isfinite(terms['total']),
                            'Nonfinite captured training objective or trajectory')
        terms['total'].backward()
        norm = None
        if settings['grad_clip']:
            norm = torch.nn.utils.clip_grad_norm_(pair.parameters(), settings['grad_clip'],
                                                  error_if_nonfinite=False, foreach=True)
            torch._assert_async(torch.isfinite(norm), 'Nonfinite captured training gradient')
        optimizer.step()
        return terms, norm

    stream = torch.cuda.Stream(device=moving.device)
    stream.wait_stream(torch.cuda.current_stream(moving.device))
    graph = None
    warmup = min(3, settings['steps'])
    setup_seconds = 0.
    with torch.cuda.stream(stream):
        for step in range(1, settings['steps'] + 1):
            inputs.reset_offsets(step)
            lr_factor = learning_rate_factor(settings, step)
            rate.fill_(settings['lr'] * lr_factor)
            weight = scheduled_local_transport_weight(objective, step)
            coefficient = registration_weight(settings, objective['registration']['weight'], step)
            local_transport_weight.fill_(weight)
            registration_coefficient.fill_(coefficient)
            if step <= warmup:
                started = time.perf_counter()
                terms, norm = update()
                stream.synchronize()
                setup_seconds += time.perf_counter() - started
            else:
                if graph is None:
                    started = time.perf_counter()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        captured_output = update()
                    inputs.reset_offsets(step)
                    stream.synchronize()
                    setup_seconds += time.perf_counter() - started
                graph.replay()
                terms, norm = captured_output
            if checkpoint_callback is not None:
                checkpoint_callback(step, pair)
            if report is not None and (step % settings['log_every'] == 0 or step == settings['steps']):
                report({'step': step, 'captured_setup_seconds': setup_seconds,
                        **{k: float(v.detach()) for k, v in terms.items()},
                        'lr': settings['lr'] * lr_factor,
                        'gradient_norm_before_clip': None if norm is None else float(norm.detach()),
                        'effective_local_transport_weight': weight, 'effective_registration_weight': coefficient})
        stream.synchronize()
    torch.cuda.current_stream(moving.device).wait_stream(stream)
    pair.zero_grad(set_to_none=True)
    pair.training_metadata['execution_timings'] = {
        'setup_warmup_capture_seconds': setup_seconds,
        'definition': 'wall time of first three updates and graph creation; included in training_seconds'}
    return pair.eval()
