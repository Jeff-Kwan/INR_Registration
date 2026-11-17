"""Two equal time strata for spatial midpoint RK2 training."""
import torch

DEFAULT_INTEGRATOR = 'rk2'


def time_strata(integrator):
    if integrator != 'rk2':
        raise ValueError('Only two-half-step spatial midpoint RK2 training is supported')
    return 2


def sample_times(reference, count, random_timesteps, generator):
    # Reference sampler for numerical validation of the Sobol/CUDA spatial path.
    if random_timesteps:
        return tuple(i / count + (1 / count) * torch.rand(
            (), device=reference.device, dtype=reference.dtype, generator=generator)
            for i in range(count))
    return tuple(reference.new_tensor((i + .5) / count) for i in range(count))


def integration_metadata(integrator, random_timesteps):
    count = time_strata(integrator)
    return {'training_integrator': 'two_half_steps_spatial_midpoint_rk2',
            'training_time_strata': count, 'training_time_step': 1 / count,
            'training_time_sampling': ('uniform_per_stratum' if random_timesteps else 'stratum_midpoints')}
