"""Canonical full-gradient NCC/PD-LK/GW-NCC training with captured AdamW."""
import json
import time
import torch

from medical.models.siren import PairedSiren
from medical.models.siren_functional import siren_parameters
from medical.registration.symmetric_registration import SymmetricPairRegistration
from medical.registration.optimization import learning_rate_factor, registration_weight, pdlk_weight as scheduled_pdlk_weight
from medical.registration.ncc import ncc_options
from medical.registration.captured_objective import joint_loss
from medical.registration.temporal_graph_sampling import SobolTimeInputs
from medical.registration.training_integrators import integration_metadata

_LAST_COMPILE_SIGNATURE = None


def fit_pair(moving, fixed, config, *, report=None, affine=None, moving_mask=None,
             fixed_mask=None, checkpoint_callback=None, allow_ablation_mse=False):
    from medical.configuration import validate_config
    validate_config(config, allow_ablation_mse=allow_ablation_mse)
    settings, objective = config['training'], config['objective']
    if (not moving.is_cuda or moving.dtype != torch.float32 or fixed.dtype != moving.dtype
            or fixed.device != moving.device):
        raise ValueError('Canonical captured training requires CUDA FP32 endpoint images')
    torch.manual_seed(config['runtime']['seed'])
    torch.set_float32_matmul_precision('highest')
    pair = PairedSiren(3, **config['model']).to(moving.device)
    pair.training_metadata = {
        'regularizer': 'pdlk', 'regularizer_weight': objective['weight'],
        'regularizer_weight_schedule': objective['weight_schedule'],
        'endpoint_objective': objective['endpoint'], 'endpoint_weight': objective['endpoint_weight'],
        'endpoint_sampling': 'shared_registration_crop', 'spatial_units': 'voxel',
        **integration_metadata('rk2', True),
        'mask_role': 'crop_center_only', 'execution': 'captured',
        'matmul_precision': torch.get_float32_matmul_precision(),
        'sampling': 'native spatial CUDA RNG; scrambled2D Sobol temporal quadrature',
        'training_time_sampling': 'scrambled_sobol_per_half',
        'temporal_sampling': 'sobol', 'query_geometry': 'orthogonal',
        'registration_similarity': 'gw_ncc',
        'registration_weight_schedule': settings['registration_weight_schedule'],
        'parameter_l2': objective['parameter_l2'],
        'brightness_weight': objective['brightness_weight'], 'differential_weight': objective['c'],
        'optimization': {'optimizer': settings['optimizer'], 'lr_schedule': settings['lr_schedule']}}
    branch = SymmetricPairRegistration(
        moving, fixed, affine, seed=config['runtime']['seed'], c=objective['c'], eta=objective['eta'],
        compile_derivatives=True, training_integrator='rk2', brightness_weight=objective['brightness_weight'],
        **{k: v for k, v in objective['registration'].items() if k != 'weight'})
    branch.crop_center_sites = []
    for image, mask in ((moving, moving_mask), (fixed, fixed_mask)):
        if mask is not None:
            if mask.shape != image.shape or mask.dtype != torch.bool or mask.device != image.device:
                raise ValueError('Invalid crop-center mask')
            sites = mask.nonzero().to(image.dtype)
            if not len(sites):
                raise ValueError('Empty crop-center mask')
            branch.crop_center_sites.append(sites)
    rate = torch.tensor(settings['lr'], device=moving.device)
    optimizer = torch.optim.AdamW(pair.parameters(), lr=rate,
                                  weight_decay=settings['optimizer']['weight_decay'],
                                  fused=True, capturable=True)
    inputs = SobolTimeInputs(branch, settings['batch_size'], config['runtime']['seed'], settings['steps'])
    source_parameters, omegas = siren_parameters(pair.source)
    target_parameters, target_omegas = siren_parameters(pair.target)
    if omegas != target_omegas:
        raise ValueError('Endpoint frequencies must match')
    pdlk_weight = torch.zeros((), device=moving.device)
    registration_coefficient = torch.zeros((), device=moving.device)
    # Reuse compiled code for equal geometries, with fresh parameters and optimizer state.
    global _LAST_COMPILE_SIGNATURE
    signature = (tuple(moving.shape), str(moving.dtype), str(moving.device), json.dumps(config, sort_keys=True))
    if signature != _LAST_COMPILE_SIGNATURE:
        torch._dynamo.reset_code(joint_loss.__code__)
        _LAST_COMPILE_SIGNATURE = signature
    compiled = torch.compile(joint_loss, fullgraph=True, dynamic=False, options={'triton.cudagraphs': False})
    options_ncc = ncc_options(objective['ncc'])

    def update(tag):
        data = inputs.draw(tag)
        pair.zero_grad(set_to_none=False)
        terms, finite = compiled(
            source_parameters, target_parameters, omegas, data['coordinates'], data['coordinates'],
            data['source_truth'], data['target_truth'], data['queries'], branch.spatial_scale,
            (data['forward_positions'], data['reverse_positions']), moving, fixed, data['origin'],
            branch.domain_extent, branch.extent, data['times'], tuple(branch.sigma), branch.grid_size,
            branch.c, branch.eta, pdlk_weight, registration_coefficient, options_ncc,
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
    graphs, captured_outputs = {}, {}
    warmup = min(3, settings['steps'])
    setup_seconds = 0.
    with torch.cuda.stream(stream):
        for step in range(1, settings['steps'] + 1):
            tag = inputs.choices[step - 1]
            inputs.reset_offsets(tag, step)
            lr_factor = learning_rate_factor(settings, step)
            rate.fill_(settings['lr'] * lr_factor)
            weight = scheduled_pdlk_weight(objective, step)
            coefficient = registration_weight(settings, objective['registration']['weight'], step)
            pdlk_weight.fill_(weight)
            registration_coefficient.fill_(coefficient)
            if step <= warmup:
                started = time.perf_counter()
                terms, norm = update(tag)
                stream.synchronize()
                setup_seconds += time.perf_counter() - started
            else:
                if tag not in graphs:
                    started = time.perf_counter()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        captured_outputs[tag] = update(tag)
                    graphs[tag] = graph
                    inputs.reset_offsets(tag, step)
                    stream.synchronize()
                    setup_seconds += time.perf_counter() - started
                graphs[tag].replay()
                terms, norm = captured_outputs[tag]
            if checkpoint_callback is not None:
                checkpoint_callback(step, pair)
            if report is not None and (step % settings['log_every'] == 0 or step == settings['steps']):
                report({'step': step, 'captured_setup_seconds': setup_seconds,
                        **{k: float(v.detach()) for k, v in terms.items()},
                        'lr': settings['lr'] * lr_factor,
                        'gradient_norm_before_clip': None if norm is None else float(norm.detach()),
                        'effective_pdlk_weight': weight, 'effective_registration_weight': coefficient})
        stream.synchronize()
    torch.cuda.current_stream(moving.device).wait_stream(stream)
    pair.zero_grad(set_to_none=True)
    pair.training_metadata['execution_timings'] = {
        'setup_warmup_capture_seconds': setup_seconds,
        'definition': 'wall time of first three updates and graph creation; included in training_seconds'}
    return pair.eval()
