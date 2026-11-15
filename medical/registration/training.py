"""Paired SIREN training with live shared-grid PD-LK and registration NCC."""
import math

import torch
from medical.models.siren import PairedSiren, coordinates_from_indices
from medical.registration.objective import TrainingBatch, endpoint_loss, scheduled_weight
from medical.registration.symmetric_registration import SymmetricPairRegistration, sample_crop_indices
from medical.registration.optimization import make_optimizers, learning_rate_factor, objective_factor, registration_weight


def shared_particle_batch(branch, moving, fixed):
    """Reuse one continuous set at both endpoints and in both directions."""
    from medical.registration.image_sampling import sample_points
    # Consume the usual two draws, retaining the same forward-particle RNG stream
    # as the separate-points control; only the first set is used in tied mode.
    points = branch.sample_positions()[0]
    coordinates = 2 * points / branch.extent - 1
    batch = TrainingBatch(coordinates, sample_points(moving, points),
                          sample_points(fixed, points), target_coordinates=coordinates)
    return batch, (points, points)


def _graph_endpoint_samples(moving, fixed, count):
    """Replay the native coordinate transform and gathers for a fixed-size batch."""
    static_indices = torch.zeros(count, dtype=torch.long, device=moving.device)
    with torch.cuda.device(moving.device):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            coordinates = coordinates_from_indices(static_indices, moving.shape)
            batch = TrainingBatch(coordinates, moving.flatten()[static_indices],
                                  fixed.flatten()[static_indices], target_coordinates=coordinates)

    def sample(indices):
        static_indices.copy_(indices)
        graph.replay()
        return batch

    return sample


def fit_pair(moving, fixed, config, *, report=None, affine=None, moving_mask=None, fixed_mask=None, checkpoint_callback=None):
    from medical.configuration import validate_config
    validate_config(config)
    if config['runtime'].get('training_execution', 'standard') == 'captured':
        from medical.registration.captured_training import fit_pair as captured_fit
        return captured_fit(moving, fixed, config, report=report, affine=affine,
                            moving_mask=moving_mask, fixed_mask=fixed_mask,
                            checkpoint_callback=checkpoint_callback)
    settings, objective = config['training'], config['objective']
    tied = settings.get('tie_registration_points', False)
    metric_mode=objective.get('regularizer','pdlk')=='inverse_metric'
    torch.manual_seed(config['runtime']['seed'])
    pair = PairedSiren(3, **config['model']).to(moving.device)
    pair.requires_physical_units_metadata=config.get('spatial_units','voxel')=='mm'
    if pair.requires_physical_units_metadata:
        from medical.physical_units import metadata
        pair.physical_units_metadata=metadata(tuple(moving.shape),affine)
    pair.requires_working_grid_metadata='working_grid' in config
    pair.training_metadata={'regularizer':objective.get('regularizer','pdlk'),
                            'regularizer_weight':objective['weight'],
                            'endpoint_objective':objective['endpoint'],
                            'endpoint_sampling':'shared_registration_crop',
                            'spatial_units':config.get('spatial_units','working_grid' if 'working_grid' in config else 'voxel'),
                            'training_integrator':'two_half_steps_spatial_midpoint_rk2'}
    generator = torch.Generator(device=moving.device).manual_seed(config['runtime']['seed'])
    branch = SymmetricPairRegistration(moving, fixed, affine,
        seed=config['runtime']['seed'], c=objective['c'], eta=objective['eta'],
        compile_derivatives=config['runtime']['compile_derivatives'],
        regularizer=objective.get('regularizer','pdlk'),
        spatial_units=config.get('spatial_units','voxel'),
        **{k:v for k,v in objective['registration'].items() if k!='weight'})
    if 'working_grid' in config:
        from medical.working_grid import configure_crop
        configure_crop(branch,config['working_grid'])
    branch.crop_center_sites = []
    for image, mask in ((moving, moving_mask), (fixed, fixed_mask)):
        if mask is None:
            continue
        if mask.shape != image.shape or mask.device != image.device or mask.dtype != torch.bool:
            raise ValueError('Crop-center mask must be boolean and match its image grid/device')
        sites = mask.nonzero().to(image.dtype)
        if not len(sites):
            raise ValueError('Crop-center mask is empty')
        branch.crop_center_sites.append(sites)
    if branch.crop_center_sites and 'endpoint_sampling' in settings:
        raise ValueError('Mask-centered shared crops cannot use global structure endpoint sampling')
    pair.training_metadata['mask_role'] = 'crop_center_only'
    endpoint_samplers = None
    if 'endpoint_sampling' in settings:
        from medical.registration.endpoint_sampling import StructureEndpointSampler
        if affine is None:
            raise ValueError('Structure sampling requires the image affine for physical smoothing')
        endpoint_samplers = [StructureEndpointSampler(image, None, affine, settings['endpoint_sampling'])
                             for image in (moving, fixed)]
        pair.training_metadata['endpoint_sampling'] = {
            'method': 'gaussian_gradient_mixture', 'inverse_probability_weighting': False,
            'endpoints': [sampler.metadata for sampler in endpoint_samplers]}
    if tied:
        pair.training_metadata['endpoint_sampling'] = 'shared_continuous_registration_starts'
        pair.training_metadata['shared_points'] = settings['batch_size']
        pair.training_metadata['shared_both_directions'] = True
    if objective['endpoint'] == 'lncc':
        pair.training_metadata['endpoint_sampling'] = {
            'method': 'dense_random_windows', **objective['endpoint_lncc'],
            'global_local_weights': [.5, .5], 'squared_correlation': False,
            'boundary': 'windows_fully_inside_current_crop'}
    crop_sampler = None
    if 'structure_crop_sampling' in settings:
        from medical.registration.endpoint_sampling import StructureCropSampler
        if not branch.cropped:
            raise ValueError('Structure crop sampling requires cropped training')
        crop_sampler = StructureCropSampler(moving, fixed, affine,
            settings['structure_crop_sampling'], seed=config['runtime']['seed'])
        pair.training_metadata['crop_sampling'] = crop_sampler.metadata
    optimizers = make_optimizers(pair, settings)
    pair.training_metadata['optimization'] = {
        'optimizer': settings.get('optimizer', {'name': 'adam'}),
        'lr_schedule': settings.get('lr_schedule', {'name': 'exponential'}),
        'objective_warmup_steps': settings.get('objective_warmup_steps', 0),
        'objective_ramp_steps': settings.get('objective_ramp_steps', 0),
        'first_layer_lr_factor': settings.get('first_layer_lr_factor', 1.),
        'registration_weight_schedule': settings.get('registration_weight_schedule')}
    final_fraction = settings.get('final_crop_fraction')
    initial_fraction = objective['registration']['crop_fraction']
    # Keep the eager sampler and optimizer. Amortize native decoder graph setup
    # for long fixed-crop CUDA fits with constant observed images.
    graph_endpoint_samples = None
    if (moving.is_cuda and moving.dtype == torch.float32 and branch.compile_velocity
            and not branch.compile_decoder and branch.registration_similarity == 'ncc'
            and branch.gaussian_kernel == 'three_tap' and final_fraction is None
            and not moving.requires_grad and not fixed.requires_grad
            and objective['registration']['weight'] and settings['steps'] >= 100):
        branch.enable_training_decoder_graph()
        branch.compile_crop_indices = True
        if not tied and endpoint_samplers is None and objective['endpoint'] in ('mse', 'ncc'):
            graph_endpoint_samples = _graph_endpoint_samples(moving, fixed, settings['batch_size'])
    for step in range(1, settings['steps']+1):
        new_shape = None
        if final_fraction is not None:
            progress = (step - 1) / (settings['steps'] - 1)
            fraction = (1 - progress) * initial_fraction + progress * final_fraction
            shape = tuple(max(2, math.ceil(n * fraction)) for n in moving.shape)
            if shape != branch.crop_shape:
                new_shape = shape
        center = None if crop_sampler is None else crop_sampler.sample_center(moving.shape)
        indices = (sample_crop_indices(branch, settings['batch_size'], generator, crop_shape=new_shape, center_xyz=center) if branch.cropped
                   else torch.randint(moving.numel(), (settings['batch_size'],), device=moving.device, generator=generator))
        positions = None
        if tied:
            batch, positions = shared_particle_batch(branch, moving, fixed)
        elif graph_endpoint_samples is not None:
            batch = graph_endpoint_samples(indices)
        else:
            endpoint_indices = [indices, indices]
            if endpoint_samplers is not None:
                endpoint_indices = [sampler.sample(settings['batch_size'], generator) for sampler in endpoint_samplers]
            source_indices, target_indices = endpoint_indices
            source_coordinates = coordinates_from_indices(source_indices, moving.shape)
            target_coordinates = (source_coordinates if source_indices is target_indices
                                  else coordinates_from_indices(target_indices, fixed.shape))
            batch = TrainingBatch(source_coordinates,
                                  moving.flatten()[source_indices], fixed.flatten()[target_indices],
                                  target_coordinates=target_coordinates)
        if objective['endpoint'] == 'lncc':
            from medical.registration.endpoint_lncc import sample_window_batch
            batch = sample_window_batch(branch, moving, fixed, generator,
                objective['endpoint_lncc'], config['working_grid']['spacing_mm'])
        if objective['endpoint'] == 'gw_ncc':
            from medical.registration.symmetric_registration import source_grid_indices
            target_coordinates = batch.coordinates if batch.target_coordinates is None else batch.target_coordinates
            batch.source_cells = source_grid_indices((batch.coordinates+1)*branch.extent/2,
                                                     branch.origin, branch.domain_extent, 1.,
                                                     grid_size=objective.get('endpoint_gw_grid_size', 8))
            batch.target_cells = source_grid_indices((target_coordinates+1)*branch.extent/2,
                                                     branch.origin, branch.domain_extent, 1.,
                                                     grid_size=objective.get('endpoint_gw_grid_size', 8))
        terms = endpoint_loss(pair, batch, objective,
                              compile_endpoints=config['runtime']['compile_endpoints'])
        activation = objective_factor(settings, step)
        weight = scheduled_weight(objective, step, settings['steps']) * activation
        registration_coefficient = registration_weight(settings, objective['registration']['weight'], step) * activation
        checks = []
        if weight or registration_coefficient:
            registration = branch.loss(pair, positions=positions, need_pdlk=bool(weight),
                need_registration=bool(registration_coefficient), diagnostics=False, deferred_checks=checks if moving.is_cuda else None)
            name='inverse_metric' if metric_mode else 'pdlk'
            terms[name] = registration['inverse_metric' if metric_mode else 'grid_pdlk']
            terms['registration'] = registration['registration']
            terms['total'] = terms['total'] + weight*terms[name] + registration_coefficient*terms['registration']
        trajectory_checks = len(checks)
        checks.append(torch.isfinite(terms['total']))
        if not moving.is_cuda and not bool(checks[-1]):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        pair.zero_grad(set_to_none=True)
        terms['total'].backward()
        gradient_norm = None
        if settings['grad_clip']:
            gradient_norm = torch.nn.utils.clip_grad_norm_(pair.parameters(), settings['grad_clip'], error_if_nonfinite=False)
            checks.append(torch.isfinite(gradient_norm))
        status = torch.stack(checks)
        if not bool(status.all()):
            valid = status.tolist()
            if not all(valid[:trajectory_checks]):
                raise RuntimeError(f'Nonfinite bidirectional particle trajectory at step {step}')
            if not valid[trajectory_checks]:
                raise RuntimeError(f'Nonfinite loss at step {step}')
            raise RuntimeError(f'Nonfinite gradient norm at step {step}')
        lr_factor = learning_rate_factor(settings, step)
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group['lr'] = group['convergence_base_lr'] * lr_factor
            optimizer.step()
        if checkpoint_callback is not None:
            checkpoint_callback(step, pair)
        if report is not None and (step%settings['log_every']==0 or step==settings['steps']):
            report({'step':step, **({'crop_fraction': fraction, 'crop_shape_zyx': branch.crop_shape}
                                    if final_fraction is not None else {}), **{k:float(v.detach()) for k,v in terms.items()},
                    'lr': settings['lr'] * lr_factor,
                    'first_layer_lr': settings['lr'] * lr_factor * settings.get('first_layer_lr_factor', 1.),
                    'gradient_norm_before_clip': None if gradient_norm is None else float(gradient_norm.detach()),
                    'effective_pdlk_weight': weight, 'effective_registration_weight': registration_coefficient})
    pair.zero_grad(set_to_none=True)
    return pair.eval()
