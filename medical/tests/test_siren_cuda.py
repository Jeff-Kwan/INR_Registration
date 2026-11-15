"""Opt-in CUDA/Inductor parity checks: INR_TEST_CUDA=1, no optimizer steps."""
import copy
import os
import unittest

import torch

from medical.models.siren import PairedSiren
from medical.registration.objective import TrainingBatch, endpoint_loss
from medical.registration.recovery import derivative_grid
from medical.registration.symmetric_registration import SymmetricPairRegistration
from medical.tests.test_siren_derivatives import reference_derivatives


@unittest.skipUnless(os.environ.get('INR_TEST_CUDA') == '1', 'Set INR_TEST_CUDA=1 to run audited CUDA checks')
class TestSirenCUDA(unittest.TestCase):
    def assert_relative_close(self, actual, expected, tolerance=1e-3):
        actual, expected = actual.detach().double().cpu(), expected.detach().double().cpu()
        self.assertTrue(torch.isfinite(actual).all())
        error = (actual-expected).norm() / expected.norm().clamp_min(1e-8)
        self.assertLessEqual(float(error), tolerance, f'relative L2 error {float(error):.6g}')

    def pair(self, width=16, layers=2, omega=30.):
        torch.manual_seed(42)
        pair = PairedSiren(width=width, layers=layers, omega=omega).cuda()
        with torch.no_grad():
            for parameter in pair.target.parameters():
                parameter.add_(torch.randn_like(parameter)*.0001)
        return pair

    def test_compiled_derivative_values_and_all_input_gradients_against_fp64(self):
        for width, layers, omega in ((8, 1, 10.), (16, 3, 60.), (256, 4, 30.)):
            with self.subTest(width=width, layers=layers, omega=omega):
                pair = self.pair(width, layers, omega)
                coordinates = (torch.rand(3, 3, device='cuda') * 2 - 1).requires_grad_()
                time = torch.tensor(.37, device='cuda', requires_grad=True)
                scale = coordinates.new_tensor([.13, .27, .51])
                truth_pair = copy.deepcopy(pair).cpu().double()
                truth_coordinates = coordinates.detach().cpu().double().requires_grad_()
                truth_time = time.detach().cpu().double().requires_grad_()
                truth = reference_derivatives(truth_pair, truth_coordinates, truth_time, scale.cpu().double())
                weights = tuple(torch.randn_like(value) for value in truth)
                truth_gradients = torch.autograd.grad(truth, (*truth_pair.parameters(), truth_coordinates, truth_time), weights)
                for compiled in (False, True):
                    values = pair.prepare(compile_parameters=compiled).at(time).derivatives(coordinates, scale,
                                differential=True, compile_derivatives=compiled)
                    gradients = torch.autograd.grad(values, (*pair.parameters(), coordinates, time),
                                                   tuple(value.to(coordinates) for value in weights))
                    for actual, expected in zip(values, truth):
                        self.assert_relative_close(actual, expected)
                    for actual, expected in zip(gradients, truth_gradients):
                        self.assert_relative_close(actual, expected)

    def test_compiled_endpoint_modes_preserve_values_and_parameter_gradients(self):
        pair = self.pair()
        source_coordinates = torch.rand(11, 3, device='cuda', requires_grad=True)
        target_coordinates = torch.rand(11, 3, device='cuda', requires_grad=True)
        batch = TrainingBatch(source_coordinates, torch.rand(11, device='cuda'),
            torch.rand(11, device='cuda'), target_coordinates=target_coordinates)
        for mode in ('mse', 'ncc'):
            with self.subTest(mode=mode):
                eager = endpoint_loss(pair, batch, {'endpoint':mode})
                compiled = endpoint_loss(pair, batch, {'endpoint':mode}, compile_endpoints=True)
                self.assertEqual(eager.keys(), compiled.keys())
                for key in eager:
                    self.assert_relative_close(compiled[key], eager[key])
                inputs = (*pair.parameters(), source_coordinates, target_coordinates)
                expected = torch.autograd.grad(eager['total'], inputs)
                actual = torch.autograd.grad(compiled['total'], inputs)
                for value, truth in zip(actual, expected):
                    # Pearson NCC's offset-invariant bias gradient cancels to
                    # nearly zero. FP64 investigation bounds FP32 reduction
                    # roundoff here; also enforce a strict whole-VJP norm below.
                    if mode == 'ncc' and (value-truth).abs().max() <= 1e-6:
                        continue
                    self.assert_relative_close(value, truth)
                self.assert_relative_close(torch.cat([value.flatten() for value in actual]),
                                           torch.cat([value.flatten() for value in expected]))

    def test_compiled_frozen_grid_tail_and_times(self):
        pair = self.pair()
        shape = (3, 4, 5)
        for time in (0., .37, 1.):
            expected = derivative_grid(pair, shape, time, chunk=11, compile_derivatives=False)
            actual = derivative_grid(pair, shape, time, chunk=11, compile_derivatives=True)
            for value, truth in zip(actual, expected):
                self.assert_relative_close(value, truth)

    def test_default_frozen_chunk_preserves_large_chunk_values(self):
        pair = self.pair(width=256, layers=4)
        shape = (41, 43, 47)  # Full 65536-point chunk plus a nonempty tail.
        expected = derivative_grid(pair, shape, .37, chunk=262144, compile_derivatives=False)
        actual = derivative_grid(pair, shape, .37, compile_derivatives=True)
        for value, truth in zip(actual, expected):
            self.assert_relative_close(value, truth)

    def test_compiled_velocity_recompute_and_live_trajectories(self):
        pair = self.pair(width=8, layers=2, omega=10.)
        image = torch.rand(9, 9, 9, device='cuda')
        initial = SymmetricPairRegistration(image, image.flip(0), torch.eye(4), grid_size=2,
            particles=32,  pairs_per_cell=2, chunk=12,
            gaussian_kernel='three_tap', compile_derivatives=False, recompute=False)
        for compiled_derivatives, compiled_velocity, recompute in (
                (False, False, False), (True, False, False), (True, True, False), (True, True, True)):
            branch = copy.deepcopy(initial)
            branch.compile_derivatives = compiled_derivatives
            branch.compile_velocity = compiled_velocity
            branch.recompute = recompute
            terms = branch.loss(pair)
            loss = .1 * terms['grid_pdlk'] + .01 * terms['registration']
            gradients = torch.autograd.grad(loss, tuple(pair.parameters()))
            if not compiled_derivatives:
                expected_terms, expected_gradients = terms, gradients
            else:
                self.assert_relative_close(terms['grid_pdlk'], expected_terms['grid_pdlk'])
                self.assert_relative_close(terms['registration'], expected_terms['registration'])
                for actual, expected in zip(gradients, expected_gradients):
                    self.assert_relative_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
