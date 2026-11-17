"""Selected unmodified SITReg metric methods, MIT, copyright Joel Honkamaa 2023.

Source: honkamj/SITReg a665a46a3553b26ade6ff5f57815e5e8ec43a948.
See LICENSE.SITReg.txt. File I/O and model execution are deliberately excluded.
Only the determinant dispatcher is specialized to the supported 3D case.
"""
from typing import Callable, Mapping, Sequence
from composable_mapping import (CoordinateSystem, DataFormat, GridComposableMapping,
    LinearInterpolator, NearestInterpolator, mappable, samplable_volume)
from numpy import ones as np_ones
from numpy.ma import masked_invalid as np_masked_invalid
from surface_distance import compute_robust_hausdorff, compute_surface_distances
from torch import (Generator, Tensor, cat, float64, rand, zeros, eye)
from torch import bool as torch_bool
from torch import device as torch_device

def estimate_spatial_derivatives(
    mapping: Callable[[Tensor], Tensor], points: Tensor, perturbation: float = 1e-7
) -> Tensor:
    """Estimate spatial derivatives at given points using small perturbations
    along each dimension

    Args:
        mapping: Mapping for which to estimate derivatives
        points: Interpolation locations, Tensor with shape (batch_size, n_dims, *target_shape)
        perturbation: Size of perturbation used for estimation

    Returns:
        Tensor with shape (batch_size, *channel_dims, n_dims, *target_shape)
    """
    n_dims = points.size(1)
    perturbations = eye(n_dims, device=points.device, dtype=points.dtype) * perturbation
    target_shape = points.shape[2:]
    points_perturbed = (
        points[:, :, None, ...]
        + perturbations[
            (
                None,
                ...,
            )
            + (None,) * len(target_shape)
        ]
    )
    perturbed_values = mapping(points_perturbed)
    non_perturbed_values = mapping(points)
    gradients = (
        perturbed_values
        - non_perturbed_values[(...,) + (None,) + len(target_shape) * (slice(None),)]
    ) / perturbation
    return gradients

def _calculate_determinant_3d(matrix: Tensor) -> Tensor:
    return (
        matrix[:, 0, 0] * (matrix[:, 1, 1] * matrix[:, 2, 2] - matrix[:, 1, 2] * matrix[:, 2, 1])
        + matrix[:, 0, 1] * (matrix[:, 1, 2] * matrix[:, 2, 0] - matrix[:, 1, 0] * matrix[:, 2, 2])
        + matrix[:, 0, 2] * (matrix[:, 1, 0] * matrix[:, 2, 1] - matrix[:, 1, 1] * matrix[:, 2, 0])
    )

def calculate_determinant(matrix):
    return _calculate_determinant_3d(matrix)[:, None]

class SITRegMetrics:
    def __init__(self, labels, *, samples=1000000, seed=8524561973):
        self._names_to_indices_seg = labels
        self._metrics_to_compute = {'dice', 'hd95', 'sampled_determinant', 'inverse_consistency'}
        self._n_jacobian_samples = samples
        self._jacobian_sampling_seed = seed

    def _compute_inverse_consistency_metrics(
            self,
            forward_displacement_field: Tensor,
            inverse_displacement_field: Tensor,
            prefix: str,
        ) -> dict[str, int | float]:
            if "inverse_consistency" not in self._metrics_to_compute:
                return {}
            forward_displacement_field = forward_displacement_field[None]
            inverse_displacement_field = inverse_displacement_field[None]
            coordinate_system = CoordinateSystem.centered_normalized(
                spatial_shape=forward_displacement_field.shape[2:],
                dtype=forward_displacement_field.dtype,
                device=forward_displacement_field.device,
            )
            forward_mapping = samplable_volume(
                forward_displacement_field,
                coordinate_system=coordinate_system,
                sampler=LinearInterpolator(),
                data_format=DataFormat.voxel_displacements(),
            )
            inverse_mapping = samplable_volume(
                inverse_displacement_field,
                coordinate_system=coordinate_system,
                sampler=LinearInterpolator(),
                data_format=DataFormat.voxel_displacements(),
            )
            forward_composition_ddf, forward_composition_mask = (
                (forward_mapping @ inverse_mapping)
                .sample(DataFormat.voxel_displacements())
                .generate()
            )
            assert forward_composition_mask is not None
            forward_composition_n_voxels = forward_composition_mask.sum()
            forward_composition_ddf_masked = (
                forward_composition_ddf * forward_composition_mask
            )
            return {
                f"{prefix}inverse_consistency_mse": forward_composition_ddf.square()
                .mean()
                .item(),
                f"{prefix}inverse_consistency_mse_masked": (
                    forward_composition_ddf_masked.square().sum()
                    / (3 * forward_composition_n_voxels)
                ).item(),
                f"{prefix}inverse_consistency_mae": forward_composition_ddf.abs()
                .mean()
                .item(),
                f"{prefix}inverse_consistency_mae_masked": (
                    forward_composition_ddf_masked.abs().sum()
                    / (3 * forward_composition_n_voxels)
                ).item(),
                f"{prefix}inverse_consistency_max": forward_composition_ddf.abs()
                .max()
                .item(),
                f"{prefix}inverse_consistency_max_masked": forward_composition_ddf_masked.abs()
                .max()
                .item(),
            }

    def _compute_sampled_determinant_metrics(
            self,
            mapping: GridComposableMapping,
            prefix: str,
            device: torch_device | None,
        ) -> dict[str, int | float]:
            if "sampled_determinant" not in self._metrics_to_compute:
                return {}
            if self._n_jacobian_samples is None or self._jacobian_sampling_seed is None:
                raise ValueError(
                    "Number of Jacobian samples and Jacobian sampling seed "
                    "is required for sampled determinant metrics."
                )
            mapping = mapping.cast(dtype=float64)
            n_dims = len(mapping.coordinate_system.spatial_shape)
            grid = mapping.coordinate_system.grid.generate_values()
            bounds = cat(
                [
                    grid.amin(dim=list(range(2, grid.ndim))),
                    grid.amax(dim=list(range(2, grid.ndim))),
                ],
                dim=0,
            ).to(float64)
            generator = Generator(device=device).manual_seed(self._jacobian_sampling_seed)
            evaluation_points = (
                rand(
                    size=(1, self._n_jacobian_samples, n_dims),
                    device=device,
                    dtype=float64,
                    generator=generator,
                )
                * (bounds[1] - bounds[0])
                + bounds[0]
            ).permute((0, 2, 1))
            jacobian_matrices = estimate_spatial_derivatives(
                mapping=lambda x: mapping(mappable(x)).generate_values(),
                points=evaluation_points,
                perturbation=mapping.coordinate_system.grid_spacing()
                .reshape(-1, n_dims)[0, 0]
                .item()
                * 1e-7,
            )
            determinants = calculate_determinant(jacobian_matrices)
            n_neg_det = int((determinants < 0).sum())
            det_std = float(determinants.std())
            n_voxels = int(evaluation_points.size(2))
            return {
                f"{prefix}n_negative_determinants": n_neg_det,
                f"{prefix}proportion_negative_determinants": (n_neg_det / n_voxels),
                f"{prefix}det_std": det_std,
            }

    def _compute_segmentation_metrics(
            self,
            mask_1_seg: Tensor,
            mask_2_seg: Tensor,
            prefix: str,
        ) -> dict[str, float]:
            metrics: dict[str, float] = {}
            if "dice" in self._metrics_to_compute or "hd95" in self._metrics_to_compute:
                seg_dice_metrics, seg_hd95_metrics = self._organ_segmentation_metrics(
                    organs_mask_1=mask_1_seg,
                    organs_mask_2=mask_2_seg,
                    names_to_indices=self._names_to_indices_seg,
                    prefix=prefix,
                )
                if "dice" in self._metrics_to_compute:
                    metrics[f"{prefix}dice"] = float(
                        np_masked_invalid(list(seg_dice_metrics.values())).mean()
                    )
                if "hd95" in self._metrics_to_compute:
                    metrics[f"{prefix}hd95"] = float(
                        np_masked_invalid(list(seg_hd95_metrics.values())).mean()
                    )
                metrics.update(seg_dice_metrics)
                metrics.update(seg_hd95_metrics)
            return metrics

    def _transform_mask(self, mask: Tensor, displacement_field: Tensor) -> Tensor:
            if mask is None:
                return None
            mask = mask[None]
            displacement_field = displacement_field[None]
            coordinate_system = CoordinateSystem.centered_normalized(
                spatial_shape=mask.shape[2:],
                dtype=mask.dtype,
                device=mask.device,
            )
            mask_mapping = samplable_volume(
                mask,
                coordinate_system=coordinate_system,
                sampler=NearestInterpolator(),
            )
            deformation = samplable_volume(
                displacement_field,
                coordinate_system=coordinate_system,
                sampler=LinearInterpolator(),
                data_format=DataFormat.voxel_displacements(),
            )
            transformed_mask_sampled = (mask_mapping @ deformation).sample()
            transformed_mask = transformed_mask_sampled.generate_values().round()
            transformed_mask_mask = transformed_mask_sampled.generate_mask().bool()
            masked_transformed_mask = transformed_mask[0] * transformed_mask_mask[0]
            return masked_transformed_mask

    def _organ_segmentation_metrics(
            self,
            organs_mask_1: Tensor,
            organs_mask_2: Tensor,
            names_to_indices: Mapping[str, Sequence[int]],
            prefix: str,
        ) -> tuple[dict[str, float], dict[str, float]]:
            dice_metrics: dict[str, float] = {}
            hd95_metrics: dict[str, float] = {}
            for name, indices in names_to_indices.items():
                organ_mask_1 = zeros(
                    organs_mask_1.shape, dtype=torch_bool, device=organs_mask_1.device
                )
                organ_mask_2 = zeros(
                    organs_mask_2.shape, dtype=torch_bool, device=organs_mask_2.device
                )
                for index in indices:
                    organ_mask_1 |= organs_mask_1 == index
                    organ_mask_2 |= organs_mask_2 == index
                if "dice" in self._metrics_to_compute:
                    organ_dice = self._dice(organ_mask_1, organ_mask_2)
                    dice_metrics[f"{prefix}{name}_dice"] = organ_dice
                if "hd95" in self._metrics_to_compute:
                    organ_hd95 = compute_robust_hausdorff(
                        compute_surface_distances(
                            organ_mask_2[0].cpu().numpy(),
                            organ_mask_1[0].cpu().numpy(),
                            np_ones(3),
                        ),
                        95.0,
                    )
                    hd95_metrics[f"{prefix}{name}_hd95"] = organ_hd95
            return dice_metrics, hd95_metrics

    @staticmethod
    def _dice(mask_1: Tensor, mask_2: Tensor) -> float:
            intersection = (mask_1 & mask_2).sum()
            union = mask_1.sum() + mask_2.sum()
            return (2 * intersection / union).item()
