"""Cell-centred paired queries and differentiable bidirectional trajectories.

Two RK2 fields per active minibatch; no optimizer or inference
recovery here. Spatial derivatives and velocities retain native-voxel units.
"""
from functools import lru_cache, partial
import math

import torch
from torch.utils.checkpoint import checkpoint

from medical.models.siren_functional import coordinate_scale
from medical.registration.derivatives import _differential_chain, _compiled_analytical

from medical.registration.shared_grid import (
    SharedGridGeometry, smooth_velocity)
from medical.registration.geometry import identity_grid, normalized_grid, sample
from medical.registration.image_sampling import sample_points
from medical.registration.ncc import correlation
from medical.registration.training_integrators import DEFAULT_INTEGRATOR, time_strata, sample_times

GW_NCC_GRID_SIZE = 8


def sample_crop_indices(branch, count, generator, *, crop_shape=None, center_xyz=None):
    """Set one crop origin and sample native voxel indices inside that same box."""
    shape = branch.moving.shape
    crop = branch.crop_shape if crop_shape is None else tuple(crop_shape)
    if len(crop) != 3 or any(type(n) is not int or n < 2 or n > s for n, s in zip(crop, shape)):
        raise ValueError('Crop shape must fit the three-dimensional image')
    device = branch.moving.device
    if crop == branch.crop_shape:
        ranges, half = branch.crop_ranges, branch.crop_half
    else:
        ranges = torch.tensor([n - b + 1 for n, b in zip(shape, crop)], device=device)
        half = branch.moving.new_tensor([n - 1 for n in crop]) / 2
    sites = getattr(branch, 'crop_center_sites', ())
    if center_xyz is not None:
        if center_xyz.shape != (3,) or not torch.isfinite(center_xyz).all():
            raise ValueError('Crop center must be a finite XYZ point')
        center = center_xyz.flip(0)
        origin = (center - half).round().long().clamp_min(0)
        origin = torch.minimum(origin, ranges - 1)
    elif sites:
        if generator is not None and device.type == 'cuda' and len(sites) == 2 and max(map(len, sites)) < (1 << 28):
            # The two scalar randint calls have equal Philox increments at these
            # bounds. Draw both candidate indices at the same counter and choose
            # on-device; the next draw starts at the original stream position.
            endpoint = torch.randint(2, (1,), device=device, generator=generator)
            offset = generator.get_offset()
            i = torch.randint(len(sites[0]), (1,), device=device, generator=generator)
            generator.set_offset(offset)
            j = torch.randint(len(sites[1]), (1,), device=device, generator=generator)
            center = torch.where(endpoint == 0, sites[0][i], sites[1][j]).squeeze(0)
        else:
            endpoint = int(torch.randint(len(sites), (), device=device, generator=generator))
            active = sites[endpoint]
            center = active[torch.randint(len(active), (), device=device, generator=generator)]
        origin = (center - half).round().long().clamp_min(0)
        origin = torch.minimum(origin, ranges - 1)
    else:
        origin = (torch.rand(3, device=device, generator=generator) * ranges).long()
    xyz_origin = origin.flip(0).to(branch.moving.dtype)
    if crop_shape is None:
        branch.set_origin(xyz_origin)
    else:
        branch.set_box(xyz_origin, crop)
    local = torch.randint(math.prod(crop), (count,), device=device, generator=generator)
    if getattr(branch, 'compile_crop_indices', False):
        return _compiled_crop_indices()(local, origin, crop, tuple(shape))
    return _crop_indices(local, origin, crop, shape)


@lru_cache(maxsize=1)
def _compiled_crop_indices():
    return torch.compile(_crop_indices, fullgraph=True)


def _crop_indices(local, origin, crop, shape):
    """Map flattened crop voxel indices into the flattened full image."""
    indices = torch.zeros_like(local)
    local_stride, global_stride = 1, 1
    for axis in range(2, -1, -1):
        component = torch.div(local, local_stride, rounding_mode='floor') % crop[axis]
        indices += (component + origin[axis]) * global_stride
        local_stride *= crop[axis]
        global_stride *= shape[axis]
    return indices


def source_grid_indices(starts, origin, extent, fraction, grid_size=GW_NCC_GRID_SIZE):
    """Bin starts into the fixed central sampling region of the local box."""
    lower = origin + (1 - fraction) * .5 * extent
    return ((((starts - lower) / (fraction * extent)) * grid_size).floor().long()
            .clamp(0, grid_size - 1))


def shared_pair_velocity(g, q, hessian, mixed, c, eta, *, defer_errors=False, brightness_weight=1.):
    """Equal-weight pooled normal equations, one ridge per cell (not per sample)."""
    matrix = (brightness_weight * (g[..., :, None] * g[..., None, :])
              + c * (hessian.transpose(-1, -2) @ hessian)).mean(-3)
    rhs = -(brightness_weight * (g * q[..., None])
            + c * (hessian.transpose(-1, -2) @ mixed[..., None]).squeeze(-1)).mean(-2)
    system = matrix + eta * torch.eye(3, device=g.device, dtype=g.dtype)
    if defer_errors:
        velocity, info = torch.linalg.solve_ex(system, rhs[..., None], check_errors=False)
        # Defer the status read to the training step's finite check. This also
        # handles a faulty solve that happens to return finite entries.
        return velocity.squeeze(-1).masked_fill(info[..., None] != 0, float('nan'))
    return torch.linalg.solve(system, rhs[..., None]).squeeze(-1)


_deferred_pair_velocity = partial(shared_pair_velocity, defer_errors=True)


def shared_pair_energy(g, q, hessian, mixed, velocity, c, eta, *, brightness_weight=1.):
    """Local transport energy per cell with gradients through the shared solve."""
    brightness = q + (g * velocity[:, None]).sum(-1)
    differential = mixed + (hessian @ velocity[:, None, :, None]).squeeze(-1)
    return ((brightness_weight * brightness.square() + c * differential.square().sum(-1)).mean(-1)
            + eta * velocity.square().sum(-1))


def gaussian_weighted_ncc(original, warped, cell_indices, *, grid_size=8, sigma_cells=.75,
                          gaussian_kernel='three_tap', epsilon=1e-8):
    """Global plus Gaussian-window Pearson loss from sparse particle moments.

    The six raw sufficient statistics are scatter-added at nearest cell centres,
    then identically filtered. Smoothed particle mass weights the local losses.
    """
    original, warped = original.reshape(-1), warped.reshape(-1)
    if (type(grid_size) is not int or grid_size < 2 or not math.isfinite(sigma_cells)
            or sigma_cells <= 0 or not math.isfinite(epsilon) or epsilon <= 0):
        raise ValueError('Invalid GW-NCC grid, Gaussian width or epsilon')
    if original.shape != warped.shape or cell_indices.shape != (original.numel(), 3):
        raise ValueError('GW-NCC expects matching intensities and one xyz cell index per particle')
    if cell_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError('GW-NCC cell indices must be integer-valued')
    # Training clamps coordinates before discretization; avoid a per-direction
    # device-to-host bounds check in the hot path.
    ones = torch.ones_like(original)
    sufficient = torch.stack((ones, original, warped, original.square(), warped.square(),
                              original * warped), dim=-1)
    flat_indices = (cell_indices[:, 0] * grid_size + cell_indices[:, 1]) * grid_size + cell_indices[:, 2]
    accumulated = original.new_zeros((grid_size ** 3, 6)).scatter_add(
        0, flat_indices[:, None].expand(-1, 6), sufficient)
    moments = smooth_velocity(accumulated.reshape(grid_size, grid_size, grid_size, 6),
                              [sigma_cells] * 3, kernel=gaussian_kernel)
    weight, sum_a, sum_b, sum_aa, sum_bb, sum_ab = moments.unbind(-1)
    safe_weight = weight.clamp_min(epsilon)
    mean_a, mean_b = sum_a / safe_weight, sum_b / safe_weight
    variance_a = (sum_aa / safe_weight - mean_a.square()).clamp_min(0.)
    variance_b = (sum_bb / safe_weight - mean_b.square()).clamp_min(0.)
    covariance = sum_ab / safe_weight - mean_a * mean_b
    local_correlation = covariance / ((variance_a + epsilon) * (variance_b + epsilon)).sqrt()
    local_correlation = local_correlation.clamp(-1., 1.)
    active = weight > epsilon
    local_weight = torch.where(active, weight, torch.zeros_like(weight))
    local_loss = (local_weight * (1. - local_correlation)).sum() / local_weight.sum().clamp_min(epsilon)
    global_correlation = correlation(original, warped, epsilon).clamp(-1., 1.)
    global_loss = 1. - global_correlation
    return .5 * (global_loss + local_loss), global_loss, local_loss


def _pair_velocity_chunk(parameters, direction, omegas, coordinates, scale, c, eta,
                         return_energy=False, derivative_function=_differential_chain,
                         velocity_function=shared_pair_velocity, brightness_weight=1.):
    pairs = coordinates.shape[1]
    g, q, h, m = derivative_function(parameters, direction, omegas, coordinates.flatten(0, 2), scale)
    g, q = g.reshape(-1, pairs * 2, 3), q.reshape(-1, pairs * 2)
    h, m = h.reshape(-1, pairs * 2, 3, 3), m.reshape(-1, pairs * 2, 3)
    velocity = velocity_function(g, q, h, m, c, eta, brightness_weight=brightness_weight)
    if return_energy:
        return velocity, shared_pair_energy(g, q, h, m, velocity, c, eta, brightness_weight=brightness_weight)
    return velocity


@lru_cache(maxsize=1)
def _compiled_pair_velocity_chunk():
    return torch.compile(_pair_velocity_chunk, fullgraph=True)


class SymmetricPairRegistration(SharedGridGeometry):
    def __init__(self, moving, fixed, affine, *, crop_fraction=None,
                 particle_sampling_fraction=.8, random_timesteps=False, pair_offset_fraction=.2,
                 pairs_per_cell=2, seed=42, query_geometry='orthogonal',
                 registration_similarity='gw_ncc',
                 gw_ncc_sigma_cells=.75, gw_ncc_kernel='three_tap',
                 compile_derivatives=True, compile_decoder=False,
                 training_integrator=DEFAULT_INTEGRATOR,
                 **options):
        super().__init__(moving, fixed, affine, **options)
        self.compile_decoder = compile_decoder
        self.compile_derivatives = compile_derivatives
        self.spatial_scale = coordinate_scale(tuple(moving.shape), moving)
        if crop_fraction is not None and (not math.isfinite(crop_fraction) or not 0 < crop_fraction <= 1):
            raise ValueError('crop_fraction must be in (0,1] or null')
        if (type(pair_offset_fraction) not in (int, float) or not math.isfinite(pair_offset_fraction)
                or not 0 < pair_offset_fraction <= .5):
            raise ValueError('pair_offset_fraction must be in (0,.5]')
        if self.chunk < 2 * pairs_per_cell:
            raise ValueError('Pair derivative chunks must accommodate all paired sites in one cell')
        self.cropped = crop_fraction is not None
        self.particle_sampling_fraction = float(particle_sampling_fraction)
        if not 0 < self.particle_sampling_fraction <= 1:
            raise ValueError('particle_sampling_fraction must be in (0,1]')
        self.random_timesteps = random_timesteps
        self.training_integrator = training_integrator
        self.time_strata = time_strata(training_integrator)
        self.pair_offset_fraction = pair_offset_fraction
        self.crop_shape = (tuple(max(2, math.ceil(n * crop_fraction)) for n in moving.shape)
                           if crop_fraction is not None else tuple(moving.shape))
        spacing = torch.as_tensor(affine, dtype=moving.dtype, device=moving.device)[:3, :3].norm(dim=0)
        self.spacing_xyz = spacing.flip(0)
        if type(pairs_per_cell) is not int or pairs_per_cell < 1:
            raise ValueError('pairs_per_cell must be a positive integer')
        self.pairs_per_cell = pairs_per_cell
        from medical.registration.query_geometry import normal_count
        normal_count(pairs_per_cell, query_geometry)
        self.query_geometry = query_geometry
        if registration_similarity != 'gw_ncc':
            raise ValueError('Invalid registration similarity')
        self.registration_similarity = registration_similarity
        if not math.isfinite(gw_ncc_sigma_cells) or gw_ncc_sigma_cells <= 0:
            raise ValueError('gw_ncc_sigma_cells must be finite and positive')
        self.gw_ncc_sigma_cells = gw_ncc_sigma_cells
        self.gw_ncc_kernel = gw_ncc_kernel
        self.local_grid = identity_grid((self.grid_size,) * 3, moving.device).reshape(-1, 3)
        self.grid_extent = moving.new_full((3,), self.grid_size - 1)
        self.offset_generator = torch.Generator(device=moving.device).manual_seed(seed + 4000047)
        self.particle_generator = torch.Generator(device=moving.device).manual_seed(seed + 4000037)
        self.time_generator = torch.Generator(device=moving.device).manual_seed(seed + 4000057)
        self.sampled_times = sample_times(moving, self.time_strata, False, None)
        self.set_box(moving.new_zeros(3), self.crop_shape)

    def _smooth_field(self, field):
        return smooth_velocity(field, self.sigma, kernel=self.gaussian_kernel,
                               compile_smoothing=self.compile_velocity)

    def set_origin(self, origin):
        self.origin = origin
        self.centres = self.local_centres + origin
        self.coordinates = None

    def set_box(self, origin, crop_shape):
        self.crop_shape = tuple(crop_shape)
        self.crop_ranges = torch.tensor([n - c + 1 for n, c in zip(self.moving.shape, crop_shape)],
                                        device=self.moving.device)
        self.crop_half = self.moving.new_tensor([c - 1 for c in crop_shape]) / 2
        self.domain_extent = self.moving.new_tensor([n - 1 for n in reversed(self.crop_shape)])
        cell_spacing = self.domain_extent / self.grid_size
        cell_width_mm = cell_spacing * self.spacing_xyz
        self.pair_radius_mm = self.pair_offset_fraction * float(cell_width_mm.min())
        # A unit-width Gaussian in lattice coordinates has physical bandwidth
        # equal to one local field-grid sample along each axis. Its footprint
        # therefore follows the crop, image spacing and grid sampling rate.
        # Smoothing tensor axes are ZYX; field vector components are XYZ.
        self.sigma = [self.sigma_cells] * 3
        self.local_centres = (self.local_grid + .5) / self.grid_size * self.domain_extent
        self.set_origin(origin)

    def resample_queries(self):
        from medical.registration.query_geometry import normal_count, query_directions
        count = normal_count(self.pairs_per_cell, self.query_geometry)
        directions = torch.randn((self.centres.shape[0], count, 3), device=self.moving.device,
                                 dtype=self.moving.dtype, generator=self.offset_generator)
        directions = query_directions(directions, self.pairs_per_cell, self.query_geometry)
        offsets = self.pair_radius_mm * directions / self.spacing_xyz
        sites = self.centres[:, None, None] + torch.stack((offsets, -offsets), -2)
        self.coordinates = normalized_grid(sites, self.moving.shape, extent=self.extent)

    def field(self, pair, time):
        return self._field_terms(pair, time)[0]

    def _field_terms(self, pair, time, *, return_energy=False, endpoints=None, need_field=True,
                     velocity_function=shared_pair_velocity):
        if self.coordinates is None:
            raise RuntimeError('Sample the shared neighbourhood before querying fields')
        prepared = (pair.prepare(compile_parameters=self.compile_derivatives or self.compile_velocity)
                    if endpoints is None else endpoints).at(time)
        compiled_velocity = self.compile_velocity and self.coordinates.is_cuda
        function = _compiled_pair_velocity_chunk() if compiled_velocity else _pair_velocity_chunk
        derivative_function = (_compiled_analytical(True)
            if self.compile_derivatives and self.coordinates.is_cuda and not compiled_velocity
            else _differential_chain)
        omegas = prepared.omegas
        def solve(coords, parameters, direction):
            return function(parameters, direction, omegas, coords, self.spatial_scale,
                self.c, self.eta, return_energy, derivative_function=derivative_function,
                velocity_function=velocity_function, brightness_weight=self.brightness_weight)
        centres_per_chunk = self.chunk // (2 * self.pairs_per_cell)
        pieces = [checkpoint(solve, coords, prepared.parameters, prepared.direction, use_reentrant=False)
                  if self.recompute else solve(coords, prepared.parameters, prepared.direction)
                  for coords in self.coordinates.split(centres_per_chunk)]
        velocities = [piece[0] for piece in pieces] if return_energy else pieces
        energy = None
        if return_energy:
            energies = torch.cat([piece[1] for piece in pieces])
            energy = energies.mean()
        field = (self._smooth_field(torch.cat(velocities).reshape(
            self.grid_size, self.grid_size, self.grid_size, 3)) if need_field else None)
        return field, energy


    def supported(self, positions):
        return ((positions >= self.origin)
                & (positions <= self.origin + self.domain_extent)).all(-1)

    def _trajectory(self, positions, fields, reverse, *, diagnostics=True, origin=None, finite_checks=None):
        # Sampled field times represent equal strata, so dt is fixed even when
        # their within-stratum locations are stochastic.
        if len(fields) != self.time_strata:
            raise ValueError('One velocity field is required per training time stratum')
        origin = self.origin if origin is None else origin
        valid = self.supported(positions) if diagnostics else None

        def check(points, message):
            finite = torch.isfinite(points).all()
            if finite_checks is not None:
                finite_checks.append(finite)
            elif not bool(finite):
                raise RuntimeError(message)
        dt = (-1 if reverse else 1) / self.time_strata
        def velocity_at(field, query):
            local = (query - origin) / self.domain_extent * self.grid_size - .5
            return sample(field, local.reshape(-1, 1, 1, 3), padding_mode='border', extent=self.grid_extent).reshape(-1, 3)
        for field in reversed(fields) if reverse else fields:
            velocity = velocity_at(field, positions)
            midpoint = positions + .5 * dt * velocity
            check(midpoint, 'Nonfinite bidirectional particle midpoint')
            if diagnostics:
                valid = valid & self.supported(midpoint)
            velocity = velocity_at(field, midpoint)
            positions = positions + dt * velocity
            check(positions, 'Nonfinite bidirectional particle trajectory')
            if diagnostics:
                valid = valid & self.supported(positions)
        return positions, valid

    def _sample_times(self, time_generator=None):
        generator = self.time_generator if time_generator is None else time_generator
        self.sampled_times = sample_times(self.moving, self.time_strata, self.random_timesteps, generator)
        return self.sampled_times

    def _time_diagnostics(self):
        return {'t_early': self.sampled_times[0], 't_late': self.sampled_times[1]}

    def _fields(self, pair, time_generator=None):
        times = self._sample_times(time_generator)
        endpoints = pair.prepare(compile_parameters=self.compile_derivatives or self.compile_velocity)
        return tuple(self._field_terms(pair, t, endpoints=endpoints)[0] for t in times)

    def integrate(self, pair, positions, *, reverse=False):
        self.resample_queries()
        return self._trajectory(positions, self._fields(pair), reverse)[0]

    def _direction_loss(self, starts, fields, reverse, *, diagnostics=True, origin=None, finite_checks=None):
        source, target = (self.fixed, self.moving) if reverse else (self.moving, self.fixed)
        if self.compile_decoder and starts.is_cuda:
            from medical.registration.compiled_decoder import compiled_box_decoder
            terms, finite = compiled_box_decoder()(starts, fields, source, target,
                self.origin, self.domain_extent, self.extent, self.grid_size, reverse,
                integrator=self.training_integrator,
                particle_sampling_fraction=self.particle_sampling_fraction,
                gw_ncc_sigma_cells=self.gw_ncc_sigma_cells, gw_ncc_kernel=self.gw_ncc_kernel)
            if finite_checks is not None:
                finite_checks.append(finite)
            elif not bool(finite):
                raise RuntimeError('Nonfinite bidirectional particle trajectory')
            return terms
        endpoints, supported = self._trajectory(
            starts, fields, reverse, diagnostics=diagnostics, origin=origin, finite_checks=finite_checks)
        return self._score_direction(starts, endpoints, reverse, supported, diagnostics=diagnostics)

    def _score_direction(self, starts, endpoints, reverse, supported=None, *, diagnostics=True):
        """Score every transported particle against the observed endpoint images."""
        source, target = (self.fixed, self.moving) if reverse else (self.moving, self.fixed)
        original = sample_points(source, starts, extent=self.extent)
        warped = sample_points(target, endpoints, extent=self.extent)
        zero = original.new_zeros(())
        mse = (original-warped).square().mean()
        cell_indices = source_grid_indices(starts, self.origin, self.domain_extent,
                                           self.particle_sampling_fraction)
        similarity, global_ncc_loss, local_ncc_loss = gaussian_weighted_ncc(
            original, warped, cell_indices, sigma_cells=self.gw_ncc_sigma_cells,
            gaussian_kernel=self.gw_ncc_kernel)
        # Image-border interpolation has zero outward coordinate derivative;
        # retain a live full-image boundary penalty for escaping trajectories.
        unit = endpoints / self.extent
        boundary = ((-unit).clamp_min(0)+(unit-1).clamp_min(0)).square().sum(-1).mean()
        if not diagnostics:
            return dict(similarity=similarity, boundary=boundary)
        return dict(mse=mse, similarity=similarity, global_ncc_loss=global_ncc_loss,
                    local_ncc_loss=local_ncc_loss, boundary=boundary,
                    outside=(~supported).float().mean(),
                    valid_particles=mse.new_tensor(len(starts)), skipped=zero)


    def sample_positions(self, generator=None):
        generator = self.particle_generator if generator is None else generator
        margin = (1 - self.particle_sampling_fraction) / 2
        low = self.origin + margin * self.domain_extent
        size = self.particle_sampling_fraction * self.domain_extent
        return tuple(low + torch.rand(self.particles, 3, device=self.moving.device,
            dtype=self.moving.dtype, generator=generator) * size for _ in range(2))

    def loss(self, pair, *, positions=None, generator=None, time_generator=None,
             need_local_transport=True, need_registration=True, diagnostics=True, deferred_checks=None):
        key='local_transport_loss'
        if not need_local_transport and not need_registration:
            zero = self.moving.new_zeros(())
            return {key:zero, 'registration_loss':zero}
        if need_registration:
            if positions is None:
                positions = self.sample_positions(generator)
            if len(positions) != 2 or any(p.ndim != 2 or p.shape[-1] != 3 or p.numel() == 0 for p in positions):
                raise ValueError('Expected nonempty forward and reverse particle arrays [N,3]')
        self.resample_queries()
        times = self._sample_times(time_generator)
        endpoints = pair.prepare(compile_parameters=self.compile_derivatives or self.compile_velocity)
        # The ordinary CUDA trainer supplies its pre-update status collector.
        # Deferral requires the full local transport energy, which observes every solve.
        defer_solve = (deferred_checks is not None and not diagnostics and need_local_transport
                       and need_registration)
        velocity_function = _deferred_pair_velocity if defer_solve else shared_pair_velocity
        paired_terms = [self._field_terms(pair, t, return_energy=need_local_transport,
                        endpoints=endpoints, need_field=need_registration, velocity_function=velocity_function) for t in times]
        fields = tuple(term[0] for term in paired_terms)
        local_transport_loss = (torch.stack([term[1] for term in paired_terms]).mean()
                     if need_local_transport else self.moving.new_zeros(()))
        if not need_registration:
            return {key:local_transport_loss,'registration_loss':local_transport_loss.new_zeros(()),
                    **self._time_diagnostics()}
        forward = self._direction_loss(positions[0], fields, False, diagnostics=diagnostics, finite_checks=deferred_checks)
        reverse = self._direction_loss(positions[1], fields, True, diagnostics=diagnostics, finite_checks=deferred_checks)
        skipped = False
        similarity = .5 * (forward['similarity'] + reverse['similarity'])
        boundary = .5 * (forward['boundary'] + reverse['boundary'])
        total = similarity + boundary
        if not diagnostics:
            return {key: local_transport_loss, 'registration_loss': total}
        image = .5 * (forward['mse'] + reverse['mse'])
        box_fraction = self.moving.new_tensor((
            self.crop_shape[2] / self.moving.shape[2],
            self.crop_shape[1] / self.moving.shape[1],
            self.crop_shape[0] / self.moving.shape[0]))
        result = dict(registration_loss=total, image_mse=image, image_similarity=similarity, boundary=boundary,
                      outside=.5 * (forward['outside'] + reverse['outside']),
                      valid_particles=torch.minimum(forward['valid_particles'], reverse['valid_particles']),
                      retained_fraction=1 - .5 * (forward['outside'] + reverse['outside']),
                      skipped=image.new_tensor(skipped), crop_x=self.origin[0],
                      crop_y=self.origin[1], crop_z=self.origin[2],
                      box_fraction_x=box_fraction[0], box_fraction_y=box_fraction[1],
                      box_fraction_z=box_fraction[2],
                      **self._time_diagnostics())
        result.update({f'{direction}_{key}': value for direction, values in
                       (('forward', forward), ('reverse', reverse)) for key, value in values.items()})
        result[key] = local_transport_loss
        return result
