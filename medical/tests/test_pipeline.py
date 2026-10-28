"""Numerical checks only; these never launch LPBA training."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

from medical.registration.flow3d import estimate_velocity, identity_grid, integrate_interval, laplacian, sample
from medical.registration.io import jacobian_determinant, label_metrics, save_volume, voxel_vector_to_world
from medical.models.siren3d import Siren3D, decode, interpolate
from medical.registration.train import (DEFAULT_CONFIG, RunETA, prepare_recovery,
                                        validate_config, recover, recovery_derivatives, evaluate,
                                        resolve_dataset_path)


class FlowTests(unittest.TestCase):
    def test_dataset_paths_resolve_from_config_with_spaces_and_repo_defaults(self):
        config = json.loads(DEFAULT_CONFIG.read_text())
        expected_root = Path(__file__).resolve().parents[2] / 'Data/LPBA40/LPBA40_delineation'
        with tempfile.TemporaryDirectory(prefix='unrelated working directory ') as cwd:
            previous = Path.cwd()
            try:
                os.chdir(cwd)
                self.assertEqual(resolve_dataset_path(config, '.').resolve(), expected_root)
            finally:
                os.chdir(previous)
        with tempfile.TemporaryDirectory(prefix='LPBA root with spaces ') as temporary:
            root = Path(temporary)
            config['dataset']['root'] = str(root)
            member = 'subject files/S02 moving.nii.gz'
            resolved = resolve_dataset_path(config, member)
            self.assertEqual(resolved, root / member)
            self.assertEqual(resolved.parent, root / 'subject files')

    def test_linear_intensity_translation_and_inverse_direction(self):
        grid = identity_grid((5, 6, 7), 'cpu')
        first = .03 * grid[..., 0]
        second = first - .03 * .2
        velocity, info = estimate_velocity(first, second, .1, ridge=1e-9,
                                            iterations=200, tolerance=1e-5)
        torch.testing.assert_close(velocity[..., 0], torch.full_like(first, 2.), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(velocity[..., 1:], torch.zeros_like(velocity[..., 1:]))
        self.assertLess(info['relative_linear_residual'], 1e-4)
        forward = integrate_interval(grid, velocity, .1)
        inverse = integrate_interval(grid, velocity, -.1)
        torch.testing.assert_close(forward[..., 0] - grid[..., 0], torch.full_like(first, .2), atol=1e-4, rtol=1e-4)
        warped = sample(first, inverse)
        torch.testing.assert_close(warped[..., 1:], second[..., 1:], atol=1e-6, rtol=1e-5)

    def test_laplacian_zero_flux_and_zero_motion(self):
        field = torch.randn(4, 5, 6, 3, dtype=torch.float64)
        torch.testing.assert_close(laplacian(field).sum((0, 1, 2)), torch.zeros(3, dtype=torch.float64), atol=1e-12, rtol=0)
        self.assertGreaterEqual(float((field * laplacian(field)).sum()), 0)
        image = torch.rand(4, 5, 6)
        velocity, info = estimate_velocity(image, image, .1)
        self.assertEqual(info['iterations'], 0)
        torch.testing.assert_close(velocity, torch.zeros_like(velocity))


class ModelAndMetricTests(unittest.TestCase):
    def test_interpolation_gradients_and_single_pass_decode(self):
        torch.manual_seed(4)
        a, b = Siren3D(8, 2), Siren3D(8, 2)
        coords = torch.rand(9, 3)
        torch.testing.assert_close(interpolate(a, b, coords, 0.), a(coords))
        torch.testing.assert_close(interpolate(a, b, coords, 1.), b(coords))
        interpolate(a, b, coords, .4).square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in list(a.parameters()) + list(b.parameters())))
        np.testing.assert_allclose(decode(a, b, (3, 4, 5), .4), decode(a, b, (3, 4, 5), .4, 7), atol=1e-7)

    def test_inference_reuses_activations_without_changing_outputs(self):
        torch.manual_seed(12)
        source, target = Siren3D(8, 2), Siren3D(8, 2)
        coordinates = torch.rand(19, 3)
        expected = interpolate(source, target, coordinates, .4)
        with torch.no_grad():
            actual = interpolate(source, target, coordinates, .4)
        torch.testing.assert_close(actual, expected)
        expected.sum().backward()
        self.assertTrue(any(p.grad is not None for p in source.parameters()))
        saved = {key: value.clone() for key, value in source.state_dict().items()}
        prepare_recovery(source, target)
        self.assertTrue(all(p.grad is None for p in list(source.parameters()) + list(target.parameters())))
        for key, value in source.state_dict().items():
            torch.testing.assert_close(value, saved[key])

    def test_physical_units_jacobian_and_label_distances(self):
        grid = identity_grid((5, 6, 7), 'cpu').numpy()
        affine = np.diag([2., 3., 4., 1.])
        displacement = np.ones_like(grid)
        np.testing.assert_array_equal(voxel_vector_to_world(displacement, affine)[0, 0, 0], [2., 3., 4.])
        np.testing.assert_allclose(jacobian_determinant(displacement), 1.)
        stretch = np.zeros_like(grid)
        stretch[..., 0] = .2 * grid[..., 0]
        np.testing.assert_allclose(jacobian_determinant(stretch), 1.2, atol=1e-6)
        a, b = np.zeros((5, 6, 7), np.int16), np.zeros((5, 6, 7), np.int16)
        a[1, 2, 3], b[2, 2, 3] = 1, 1
        metrics, _ = label_metrics(a, b, (2., 3., 4.))
        self.assertEqual(metrics['dice'], 0.)
        self.assertEqual(metrics['hd95_mm'], 2.)
        missing, _ = label_metrics(a, np.zeros_like(a), (2., 3., 4.))
        self.assertIsNone(missing['hd95_mm'])
        self.assertEqual(missing['missing_labels'], [1])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'flow.nii.gz'
            save_volume(path, displacement, nib.Nifti1Image(a, affine), vector=True)
            image = nib.load(path)
            self.assertEqual(image.shape, (*a.shape, 3))
            np.testing.assert_allclose(image.affine, affine)

    def test_analytical_recovery_pipeline_and_legacy_config(self):
        config = json.loads(DEFAULT_CONFIG.read_text())
        shape = (3, 4, 5)
        source, target = Siren3D(8, 2), Siren3D(8, 2)
        target.load_state_dict(source.state_dict())
        config['recovery'].update(intervals=2, iterations=10)
        config['runtime']['inference_chunk'] = 7
        config['output'].update(save_frames=False, save_velocities=True)
        reference = nib.Nifti1Image(np.zeros(shape, np.float32), np.eye(4))
        whole = recovery_derivatives(source, target, shape, .4)
        chunked = recovery_derivatives(source, target, shape, .4, 7)
        for a, b in zip(whole, chunked):
            torch.testing.assert_close(a, b)
        for method, save_frames in [('analytical', False), ('finite', False), ('analytical', True)]:
            config['recovery']['method'] = method
            config['output']['save_frames'] = save_frames
            validate_config(config)
            eta = RunETA(1, config)
            with tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                forward, inverse, src, tgt, diagnostics = recover(
                    source, target, shape, reference, config, output, 'cpu', eta)
                torch.testing.assert_close(forward, identity_grid(shape, 'cpu'))
                torch.testing.assert_close(inverse, forward)
                np.testing.assert_allclose(src, tgt)
                self.assertEqual(len(diagnostics), 2)
                self.assertEqual(diagnostics[0]['method'], method)
                self.assertEqual(np.load(output/'velocities.npy').shape, (2, *shape, 3))
                self.assertEqual((output/'frames.npy').exists(), save_frames)
                if save_frames:
                    self.assertEqual(np.load(output/'frames.npy').shape, (3, *shape))
                self.assertEqual(eta.completed['decode'], eta.units['decode'])
        legacy = copy.deepcopy(config)
        legacy['recovery'].pop('intervals')
        legacy['recovery'].pop('method')
        legacy['recovery']['frames'] = 33
        validate_config(legacy)
        self.assertEqual(legacy['recovery']['method'], 'finite')
        self.assertEqual(legacy['recovery']['intervals'], 32)
        for value in (0, -1, 1.5, True):
            invalid = copy.deepcopy(config)
            invalid['recovery']['intervals'] = value
            with self.assertRaises(ValueError):
                validate_config(invalid)

    def test_volume_saving_preserves_metrics_and_cleans_temporary_arrays(self):
        config = json.loads(DEFAULT_CONFIG.read_text())
        config['recovery'].update(intervals=2, iterations=10)
        config['output'].update(save_frames=False, save_velocities=False)
        config['evaluation']['hd95'] = False
        shape = (3, 4, 5)
        source, target = Siren3D(8, 2), Siren3D(8, 2)
        target.load_state_dict(source.state_dict())
        reference = nib.Nifti1Image(np.zeros(shape, np.float32), np.eye(4))
        image = torch.from_numpy(decode(source, target, shape, 0.))
        labels = np.ones(shape, np.int16)
        results = []
        for enabled in (True, False):
            config['output']['save_volumes'] = enabled
            validate_config(config)
            with tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                forward, inverse, src, tgt, diagnostics = recover(
                    source, target, shape, reference, config, output, 'cpu', RunETA(1, config))
                results.append(evaluate(image, image, labels, labels, forward, inverse,
                                        src, tgt, reference, config, output))
                self.assertEqual(len(list(output.glob('*.nii.gz'))), 9 if enabled else 0)
                self.assertFalse(list(output.glob('*.npy')))
                self.assertFalse(list(output.glob('recovery_*')))
                self.assertTrue((output / 'flow_diagnostics.json').exists())
                self.assertTrue((output / 'per_label_metrics.json').exists())
        self.assertEqual(results[0], results[1])

    def test_configuration_and_full_run_eta(self):
        config = json.loads(DEFAULT_CONFIG.read_text())
        validate_config(config)
        self.assertGreater(config['training']['batch_size'], 0)
        self.assertEqual(config['lk']['mode'], 'pointwise')
        self.assertEqual(config['lk']['points'], 4096)
        self.assertFalse(config['lk']['differentiate_solve'])
        self.assertEqual(config['lk']['gradient_weight'], 4.)
        self.assertGreater(config['recovery']['intervals'], 0)
        self.assertTrue(config['runtime']['compile_derivatives'])
        legacy=copy.deepcopy(config)
        legacy['runtime'].pop('compile_derivatives')
        validate_config(legacy)
        self.assertTrue(legacy['runtime']['compile_derivatives'])
        invalid_compile=copy.deepcopy(config)
        invalid_compile['runtime']['compile_derivatives']='true'
        with self.assertRaises(ValueError):
            validate_config(invalid_compile)
        self.assertEqual(config['recovery']['method'], 'analytical')
        invalid = copy.deepcopy(config)
        invalid['lk']['patch_size'] = 4
        with self.assertRaises(ValueError):
            validate_config(invalid)
        eta = RunETA(2, config)
        eta.begin_pair(0)
        for stage, units in eta.units.items():
            eta.update(stage, units, units)
        seconds = sum(eta.units.values())
        expected = f'{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}'
        self.assertEqual(eta.text(), expected)


if __name__ == '__main__':
    unittest.main()
