"""Global-flow derivative inputs, units, and RK2 analytical recovery checks."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch

import torch

from cameraman.core import flow
from cameraman.core.analytical_recovery import image_derivatives, recover_2d, recover_3d
from cameraman.core.inr import Siren
from cameraman.core.synthetic import identity_grid

from medical.registration import flow3d
from medical.models.siren3d import Siren3D


class AnalyticalRecoveryTests(unittest.TestCase):
    def test_existing_run_is_never_overwritten(self):
        from cameraman.experiments.run_experiment import prepare_output_directory
        with tempfile.TemporaryDirectory(prefix='run path with spaces ') as temporary:
            output = Path(temporary) / 'new run'
            prepare_output_directory(output)
            sentinel = output / 'metrics.json'
            sentinel.write_text('original run')
            with self.assertRaises(FileExistsError):
                prepare_output_directory(output)
            self.assertEqual(sentinel.read_text(), 'original run')

    def test_runner_defaults_and_analytical_cached_evaluation(self):
        from cameraman.experiments.run_experiment import parse_args
        from cameraman.core.reference_cache import evaluate_with_cached_references
        from cameraman.core.interpolate import interpolate_weights
        with patch.object(sys, 'argv', ['run_experiment.py']):
            args = parse_args()
        self.assertEqual((args.recovery_method, args.intervals), ('analytical', 8))
        self.assertEqual(args.min_psnr, 25.)
        self.assertEqual(args.flow_tolerance, 2e-5)
        self.assertEqual(args.image, Path(__file__).resolve().parents[1] / 'data/cameraman.jpg')
        self.assertEqual(args.output, Path(__file__).resolve().parents[2] / 'cameraman/runs/pdlk_relaxed')
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder)/'recovery.json'
            config.write_text(json.dumps({'intervals': 3, 'recovery_method': 'analytical'}))
            with patch.object(sys, 'argv', ['run_experiment.py', '--config', str(config)]):
                self.assertEqual(parse_args().intervals, 3)
            with patch.object(sys, 'argv', ['run_experiment.py', '--config', str(config), '--intervals', '5']):
                self.assertEqual(parse_args().intervals, 5)
            retired = Path(folder) / 'retired.json'
            retired.write_text(json.dumps({'patch_regularizer': 'h1'}))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--config', str(retired)])
        torch.manual_seed(90)
        a, b = Siren(8, 2), Siren(8, 2)
        grid = identity_grid(8, 'cpu')
        coords = (2*grid/7-1).reshape(-1, 2)
        sequence = interpolate_weights(a, a.state_dict(), b.state_dict(), coords, 8, 3)
        options = dict(iterations=10, substeps=1)
        mask = torch.ones((8, 8), dtype=torch.bool)
        with tempfile.TemporaryDirectory() as folder:
            result = evaluate_with_cached_references(sequence, sequence[0], sequence[-1], grid,
                torch.zeros_like(grid), mask, folder, inr_models=(a, b), **options)
            _, experiments, _, tensors = result
            expected, inverse, _, _ = recover_2d(a, b, grid, **options)
            torch.testing.assert_close(tensors['D_inr_2d']['forward_map'], expected)
            torch.testing.assert_close(tensors['D_inr_2d']['inverse_map'], inverse)
            self.assertEqual(experiments['D_inr_2d']['intervals'], 8)
            self.assertEqual(experiments['D_inr_horizontal']['recovery_method'], 'analytical')
            self.assertEqual(experiments['B_oracle_2d']['intervals'], 2)
            self.assertEqual(experiments['B_oracle_2d']['recovery_method'], 'finite')

    def test_matched_configs_and_invalid_settings(self):
        from cameraman.experiments.run_experiment import parse_args
        root = Path(__file__).resolve().parents[1] / 'configs'
        standard = vars(parse_args(['--config', str(root / 'pdlk_standard.json')]))
        relaxed = vars(parse_args(['--config', str(root / 'pdlk_relaxed.json')]))
        self.assertEqual(standard['lk_affine_rho'], 0.)
        self.assertEqual(relaxed['lk_affine_rho'], .4)
        for key in ('config', 'output', 'lk_affine_rho'):
            standard.pop(key)
            relaxed.pop(key)
        self.assertEqual(standard, relaxed)
        for option, value in (('--lk-ridge', '0'), ('--lk-affine-rho', '-1'),
                              ('--lr', 'nan'), ('--lk-points', '0')):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args([option, value])

    def test_unsupported_saved_architecture_rejected_before_checkpoint_loading(self):
        from cameraman.analysis.reevaluate import reevaluate
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / 'config.json').write_text(json.dumps({'inr': 'wire'}))
            with patch('cameraman.analysis.reevaluate.load_saved', side_effect=AssertionError('must not load')):
                with self.assertRaisesRegex(ValueError, 'paired SIRENs'):
                    reevaluate(run, 3, 'cpu')

    def test_rk4_linear_field_accuracy_and_inverse(self):
        for dimensions in (2, 3):
            grid = (identity_grid(10, 'cpu') if dimensions == 2 else
                    flow3d.identity_grid((10, 10, 10), 'cpu')).double()
            velocity = .2*grid
            def integrate(method, inverse=False):
                if dimensions == 2:
                    return flow.integrate(velocity[None], grid, substeps=1, inverse=inverse, method=method)
                return flow3d.integrate_interval(grid, velocity, -1. if inverse else 1., substeps=1, method=method)
            interior = (slice(1, 6),)*dimensions
            for inverse in (False, True):
                expected = grid*torch.exp(torch.tensor(-.2 if inverse else .2, dtype=torch.float64))
                e2 = (integrate('rk2', inverse)-expected)[interior].abs().max()
                e4 = (integrate('rk4', inverse)-expected)[interior].abs().max()
                self.assertLess(float(e4), float(e2)/100)
            with self.assertRaises(ValueError):
                integrate('invalid')

    def test_existing_2d_frame_solver_matches_derivative_entry_point(self):
        torch.manual_seed(5)
        frames = torch.rand(4, 7, 9, dtype=torch.float64)
        gy, gx = torch.gradient((frames[:-1]+frames[1:])*.5, dim=(1, 2))
        for horizontal in (False, True):
            args = dict(horizontal=horizontal, iterations=50)
            old = flow.estimate_velocity(frames, **args)
            new = flow.estimate_velocity_from_derivatives(torch.stack((gx, gy), -1),
                (frames[1:]-frames[:-1])*3, **args)
            for a, b in zip(old[:2], new[:2]):
                torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)

    def test_existing_3d_frame_solver_matches_derivative_entry_point(self):
        torch.manual_seed(6)
        first, second = torch.rand(2, 5, 6, 7, dtype=torch.float64)
        gz, gy, gx = torch.gradient((first+second)*.5)
        old = flow3d.estimate_velocity(first, second, .25, iterations=50)
        new = flow3d.estimate_velocity_from_derivatives(torch.stack((gx, gy, gz), -1),
            (second-first)/.25, iterations=50)
        torch.testing.assert_close(old[0], new[0], atol=0, rtol=0)
        self.assertEqual(old[1], new[1])

    def test_derivative_query_chunking_preserves_values(self):
        torch.manual_seed(7)
        a, b = Siren(8, 2).double(), Siren(8, 2).double()
        x = torch.rand(9*11, 2, dtype=torch.float64)
        whole = image_derivatives(a, b, x, .3, (9, 11))
        chunked = image_derivatives(a, b, x, .3, (9, 11), chunk=17)
        for expected, actual in zip(whole, chunked):
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_analytical_translation_and_rk2_units(self):
        a = Siren(2, 1, omega=1.).double()
        with torch.no_grad():
            a.net[0].linear.weight.copy_(torch.diag(torch.tensor([2., 3.])))
            a.net[0].linear.bias.zero_()
            a.net[-1].weight.fill_(1.)
            a.net[-1].bias.zero_()
        b = copy.deepcopy(a)
        grid = identity_grid(13, 'cpu').double()
        velocity = torch.tensor([.2, -.1], dtype=torch.float64)
        with torch.no_grad():
            b.net[0].linear.bias.copy_(-torch.tensor([2., 3.])*velocity*2/12)
        forward, inverse, _, _ = recover_2d(a, b, grid, intervals=4, ridge=1e-12,
            iterations=1000, tolerance=1e-9)
        torch.testing.assert_close(forward-grid, velocity.expand_as(grid), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(inverse-grid, -velocity.expand_as(grid), atol=1e-5, rtol=1e-5)

    def test_streaming_3d_translation_and_query_chunks(self):
        a = Siren3D(3, 1, omega=1.).double()
        frequency = torch.tensor([2., 3., 5.], dtype=torch.float64)
        scale = torch.tensor([2/7, 2/6, 2/5], dtype=torch.float64)
        velocity = torch.tensor([.1, -.07, .03], dtype=torch.float64)
        with torch.no_grad():
            a.net[0].linear.weight.copy_(torch.diag(frequency))
            a.net[0].linear.bias.zero_()
            a.net[-1].weight.fill_(1.)
            a.net[-1].bias.zero_()
        b = copy.deepcopy(a)
        with torch.no_grad():
            b.net[0].linear.bias.copy_(-frequency*scale*velocity)
        grid = flow3d.identity_grid((6, 7, 8), 'cpu').double()
        forward, inverse, _ = recover_3d(a, b, grid, flow3d, intervals=4, chunk=41,
            ridge=1e-12, iterations=1000, tolerance=1e-9)
        torch.testing.assert_close(forward-grid, velocity.expand_as(grid), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(inverse-grid, -velocity.expand_as(grid), atol=1e-5, rtol=1e-5)
        with self.assertRaises(ValueError):
            recover_3d(a, b, grid, flow3d, chunk=-1)


if __name__ == '__main__':
    unittest.main()
