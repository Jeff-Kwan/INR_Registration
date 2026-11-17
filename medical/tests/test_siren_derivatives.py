"""Synthetic mathematical regression checks; no optimizer or saved-model recovery."""
import copy
import unittest
from unittest.mock import patch

import torch

from medical.models.siren import PairedSiren, Siren, coordinates_from_indices, endpoint_reconstruction
from medical.models.siren_functional import coordinate_scale, siren_forward
from medical.registration.derivatives import _differential_chain, _layer_curvature_chain
from medical.registration.recovery import derivative_grid
from medical.registration.objective import TrainingBatch, endpoint_loss
from medical.registration.symmetric_registration import SymmetricPairRegistration


def reference_derivatives(pair, coordinates, time, scale):
    """Independent nested autograd oracle over the ordinary module forward."""
    target = dict(pair.target.named_parameters())
    parameters = {name: torch.lerp(value, target[name], time)
                  for name, value in pair.source.named_parameters()}
    values = torch.func.functional_call(pair.source, parameters, (coordinates,)).squeeze(-1)
    gradient = torch.autograd.grad(values.sum(), coordinates, create_graph=True)[0]
    temporal = torch.stack([torch.autograd.grad(value, time, create_graph=True, retain_graph=True)[0]
                            for value in values.reshape(-1)]).reshape(values.shape)
    hessian = torch.stack([torch.autograd.grad(gradient[..., axis].sum(), coordinates,
                            create_graph=True, retain_graph=True)[0]
                          for axis in range(coordinates.shape[-1])], dim=-2)
    mixed = torch.autograd.grad(temporal.sum(), coordinates, create_graph=True)[0]
    return (gradient * scale, temporal,
            hessian * scale[:, None] * scale[None, :], mixed * scale)


def make_pair(dimensions=3, width=5, layers=2, omega=30., dtype=torch.float64):
    torch.manual_seed(42)
    pair = PairedSiren(dimensions, width=width, layers=layers, omega=omega).to(dtype=dtype)
    with torch.no_grad():
        for parameter in pair.target.parameters():
            parameter.add_(torch.randn_like(parameter) * .003)
    return pair


class TestAnalyticalSiren(unittest.TestCase):
    def test_streamed_whole_volume_psnr(self):
        pair=make_pair(width=4,layers=1)
        with torch.no_grad():
            for model in (pair.source,pair.target):
                model.net[-1].weight.zero_();model.net[-1].bias.zero_()
        image=torch.full((3,4,5),.5,dtype=torch.float64)
        scores=endpoint_reconstruction(pair,(image,torch.zeros_like(image)),chunk=11)
        self.assertAlmostEqual(scores['moving']['mse'],.25)
        self.assertAlmostEqual(scores['moving']['psnr_db'],6.020599913279624)
        self.assertTrue(scores['fixed']['perfect_reconstruction'])
        self.assertIsNone(scores['fixed']['psnr_db'])

    def test_values_and_all_input_vjps_against_independent_autograd(self):
        for dimensions, layers, omega, time_value, identical in (
                (2, 1, 10., 0., False), (3, 2, 30., .37, False),
                (3, 4, 60., 1., False), (2, 3, 3., .37, False),
                (3, 2, 30., .37, True)):
            with self.subTest(dimensions=dimensions, layers=layers, omega=omega,
                              time=time_value, identical=identical):
                pair = make_pair(dimensions, layers=layers, omega=omega)
                if identical:
                    pair.target.load_state_dict(pair.source.state_dict())
                coordinates = torch.rand(2, dimensions, 2, dtype=torch.float64).transpose(1, 2).requires_grad_()
                time = torch.tensor(time_value, dtype=torch.float64, requires_grad=True)
                scale = torch.tensor([.13, .27, .51][:dimensions], dtype=torch.float64)
                prepared = pair.prepare().at(time)
                expected = reference_derivatives(pair, coordinates, time, scale)
                weights = tuple(torch.randn_like(value) for value in expected)
                inputs = (*pair.parameters(), coordinates, time)
                references = torch.autograd.grad(expected, inputs, weights)
                for chain in (_differential_chain, _layer_curvature_chain):
                    actual = chain(prepared.parameters, prepared.direction, prepared.omegas,
                                   coordinates, scale)
                    for value, truth in zip(actual, expected):
                        torch.testing.assert_close(value, truth, rtol=1e-8, atol=1e-10)
                    torch.testing.assert_close(actual[2], actual[2].transpose(-1, -2), rtol=1e-8, atol=1e-10)
                    gradients = torch.autograd.grad(actual, inputs, weights, retain_graph=True)
                    for value, truth in zip(gradients, references):
                        torch.testing.assert_close(value, truth, rtol=1e-8, atol=1e-10)

    def test_basic_agrees_with_differential(self):
        pair = make_pair()
        coordinates = torch.rand(5, 3, dtype=torch.float64)
        prepared = pair.prepare().at(.37)
        basic = prepared.derivatives(coordinates, [.13, .27, .51], compile_derivatives=False)
        full = prepared.derivatives(coordinates, [.13, .27, .51],
                                    differential=True, compile_derivatives=False)
        for value, truth in zip(basic, full[:2]):
            torch.testing.assert_close(value, truth, rtol=1e-8, atol=1e-10)

    def test_parameter_coordinate_time_gradcheck_and_gradgradcheck(self):
        pair = make_pair(2, width=3, layers=1, omega=3.)
        endpoints = pair.prepare()
        original = tuple(value for layer in endpoints.source + endpoints.target for value in layer)
        coordinates = torch.tensor([[.17, -.23]], dtype=torch.float64, requires_grad=True)
        time = torch.tensor(.37, dtype=torch.float64, requires_grad=True)
        scale = coordinates.new_tensor([.13, .27])
        count = len(endpoints.source)
        def function(*inputs):
            first = tuple((inputs[2*i], inputs[2*i+1]) for i in range(count))
            second = tuple((inputs[2*count+2*i], inputs[2*count+2*i+1]) for i in range(count))
            parameters = tuple(tuple(torch.lerp(a, b, inputs[-1]) for a, b in zip(left, right))
                               for left, right in zip(first, second))
            direction = tuple(tuple(b-a for a, b in zip(left, right)) for left, right in zip(first, second))
            return _differential_chain(parameters, direction, endpoints.omegas, inputs[-2], scale)
        inputs = (*original, coordinates, time)
        self.assertTrue(torch.autograd.gradcheck(function, inputs, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.autograd.gradgradcheck(function, inputs, atol=1e-5, rtol=1e-4))

    def test_fused_endpoint_affines_match_module_values_and_gradients(self):
        pair = make_pair()
        endpoints = pair.prepare()
        coordinates = torch.rand(2, 3, 4, dtype=torch.float64).transpose(1, 2).requires_grad_()
        expected = pair.source(coordinates)
        actual = siren_forward(coordinates, endpoints.source, endpoints.omegas, fuse_scalar_affines=True)
        torch.testing.assert_close(actual, expected)
        inputs = (coordinates, *pair.source.parameters())
        probe = torch.randn_like(actual)
        for value, reference in zip(torch.autograd.grad(actual, inputs, probe),
                                    torch.autograd.grad(expected, inputs, probe)):
            torch.testing.assert_close(value, reference, rtol=1e-11, atol=1e-12)

    def test_forward_and_checkpoint_layout_stay_compatible(self):
        pair = make_pair()
        coordinates = torch.rand(3, 3, dtype=torch.float64)
        endpoints = pair.prepare()
        torch.testing.assert_close(siren_forward(coordinates, endpoints.source, endpoints.omegas), pair.source(coordinates))
        target = dict(pair.target.named_parameters())
        parameters = {name: torch.lerp(value, target[name], .37)
                      for name, value in pair.source.named_parameters()}
        expected = torch.func.functional_call(pair.source, parameters, (coordinates,))
        torch.testing.assert_close(pair.image(coordinates, .37), expected)
        self.assertEqual(list(pair.state_dict()), [f'{endpoint}.{name}' for endpoint in ('source', 'target')
                                                  for name, _ in pair.source.named_parameters()])
        before = pair.image(coordinates, .37)
        with torch.no_grad():
            pair.source.net[-1].bias.add_(1.)
        torch.testing.assert_close(pair.image(coordinates, .37), before + .63)

    def test_frozen_grid_and_decode_prepare_once_including_tail(self):
        pair = make_pair()
        shape = (3, 4, 5)
        coordinates = coordinates_from_indices(torch.arange(60), shape).double()
        with torch.no_grad():
            expected = pair.derivatives(coordinates, .37, shape, compile_derivatives=False)
            expected_image = pair.image(coordinates, .37).reshape(shape).float()
        with patch.object(pair, 'prepare', wraps=pair.prepare) as prepare:
            actual = derivative_grid(pair, shape, .37, chunk=11, compile_derivatives=False)
            self.assertEqual(prepare.call_count, 1)
        torch.testing.assert_close(actual[0].reshape(-1, 3), expected[0])
        torch.testing.assert_close(actual[1].flatten(), expected[1])
        with patch.object(pair, 'prepare', wraps=pair.prepare) as prepare:
            decoded = pair.decode(shape, .37, chunk=11)
            self.assertEqual(prepare.call_count, 1)
        torch.testing.assert_close(decoded, expected_image)

    def test_invalid_architecture_and_metadata_fail_early(self):
        for options in ({'omega':float('nan')}, {'omega':float('inf')}, {'width':2.5},
                        {'layers':True}, {'dimensions':True}, {'omega':0.}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Siren(**options)
        pair = make_pair()
        pair.target.net[0].omega = 31.
        with self.assertRaisesRegex(ValueError, 'omega'):
            pair.prepare()
        with self.assertRaises(ValueError):
            coordinate_scale((1, 4, 5), torch.zeros(1))
        pair = make_pair()
        with self.assertRaisesRegex(ValueError, 'Coordinates'):
            pair.image(torch.ones(3, 2, dtype=torch.float64), .37)
        with self.assertRaisesRegex(ValueError, 'Time'):
            pair.prepare().at(torch.ones(1, dtype=torch.float64))


class TestLiveRegistration(unittest.TestCase):
    def branch(self, *, recompute=False, chunk=16):
        torch.manual_seed(42)
        image = torch.rand(9, 9, 9, dtype=torch.float64)
        return SymmetricPairRegistration(image, image.flip(0), torch.eye(4), grid_size=2,
            particles=32,  pairs_per_cell=2, chunk=chunk,
            recompute=recompute, gaussian_kernel='three_tap', compile_derivatives=False)

    def test_direction_particles_use_shared_crop(self):
        branch = self.branch()
        moving, fixed = branch.sample_positions()
        self.assertTrue(branch.supported(moving).all())
        self.assertTrue(branch.supported(fixed).all())
        self.assertFalse(torch.equal(moving, fixed))

    def test_endpoint_reconstruction_uses_separate_coordinates(self):
        pair = make_pair(width=4, layers=1)
        source = torch.tensor([[.2, 0., 0.]], dtype=torch.float64)
        target = torch.tensor([[.7, 0., 0.]], dtype=torch.float64)
        batch = TrainingBatch(source, pair.source(source).flatten(), pair.target(target).flatten(),
                              target_coordinates=target)
        terms = endpoint_loss(pair, batch, {'endpoint':'ncc', 'ncc': {'moment_weight': 0.}})
        self.assertEqual(terms['source_mse'], 0)
        self.assertEqual(terms['target_mse'], 0)

    def test_recomputation_and_tail_chunks_preserve_live_solve_gradients(self):
        pair = make_pair(width=4, omega=10.)
        branch = self.branch()
        branch.resample_queries()
        time = torch.tensor(.37, dtype=torch.float64, requires_grad=True)
        for recompute, chunk in ((False, 32), (False, 12), (True, 12)):
            branch.recompute, branch.chunk = recompute, chunk
            field, energy = branch._field_terms(pair, time, return_energy=True)
            loss = field.square().mean() + energy
            gradients = torch.autograd.grad(loss, (*pair.parameters(), time))
            if recompute is False and chunk == 32:
                reference_field, reference_energy, reference_gradients = field, energy, gradients
            else:
                torch.testing.assert_close(field, reference_field)
                torch.testing.assert_close(energy, reference_energy)
                for actual, expected in zip(gradients, reference_gradients):
                    torch.testing.assert_close(actual, expected)
            self.assertTrue(all(torch.isfinite(value).all() for value in gradients))

    def test_ablation_skips_only_unused_work(self):
        pair = make_pair(width=4, omega=10.)
        branch = self.branch()
        with (patch.object(branch, '_direction_loss', side_effect=AssertionError('Unused trajectory')),
              patch.object(branch, 'sample_positions', side_effect=AssertionError('Unused positions'))):
            energy_only = branch.loss(pair, need_registration=False)
        gradients = torch.autograd.grad(energy_only['grid_pdlk'], tuple(pair.parameters()))
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
        with patch('medical.registration.symmetric_registration.shared_pair_energy',
                   side_effect=AssertionError('Unused residual energy')):
            registration_only = branch.loss(pair, need_pdlk=False)
        self.assertEqual(registration_only['grid_pdlk'], 0)
        gradients = torch.autograd.grad(registration_only['registration'], tuple(pair.parameters()))
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
        half = len(gradients)//2
        self.assertTrue(any(value.abs().sum() > 0 for value in gradients[:half]))
        self.assertTrue(any(value.abs().sum() > 0 for value in gradients[half:]))
        with patch.object(pair, 'prepare', side_effect=AssertionError('Unused derivatives')):
            self.assertEqual(branch.loss(pair, need_pdlk=False, need_registration=False)['registration'], 0)

    def test_ablation_values_and_gradients_match_full_objective(self):
        pair = make_pair(width=4, omega=10.)
        initial = self.branch()
        for key, options in (('grid_pdlk', {'need_registration':False}),
                             ('registration', {'need_pdlk':False})):
            with self.subTest(term=key):
                full = copy.deepcopy(initial).loss(pair)[key]
                ablation = copy.deepcopy(initial).loss(pair, **options)[key]
                torch.testing.assert_close(ablation, full)
                expected = torch.autograd.grad(full, tuple(pair.parameters()))
                actual = torch.autograd.grad(ablation, tuple(pair.parameters()))
                for value, truth in zip(actual, expected):
                    torch.testing.assert_close(value, truth)


if __name__ == '__main__':
    unittest.main()
