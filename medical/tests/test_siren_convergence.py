"""Synthetic checks of experimental controls; no dataset reads or recovery.

The optimizer comparison performs one explicitly authorized synthetic step.
"""
import copy
import io
import unittest

import torch

from medical.configuration import DEFAULT
from medical.models.siren import PairedSiren
from medical.registration.optimization import (
    learning_rate_factor, make_optimizers, registration_weight,
    validate_optimization,
)
from medical.tests.test_siren_derivatives import reference_derivatives


class TestConvergenceControls(unittest.TestCase):
    def test_default_initialization_and_layout_are_unchanged(self):
        torch.manual_seed(42)
        original = PairedSiren(3, width=5, layers=3, omega=30.)
        torch.manual_seed(42)
        explicit = PairedSiren(3, width=5, layers=3, omega=30., init_scale=1.,
            layer_init_scales=[1., 1., 1.], hidden_omega=30.,
            balance_first_frequencies=False, first_omega_mode='initialization',
            output_init_scale=1.)
        self.assertEqual(list(original.state_dict()), list(explicit.state_dict()))
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)

    def test_combined_frequency_controls_preserve_derivatives_and_parameter_vjps(self):
        torch.manual_seed(42)
        pair = PairedSiren(3, width=5, layers=3, omega=30., init_scale=.7,
            layer_init_scales=[1., .8, 1.1], hidden_omega=15.,
            first_omega_range=[10., 40.], first_omega_mode='parameterized',
            balance_first_frequencies=True, output_init_scale=10.).double()
        with torch.no_grad():
            for parameter in pair.target.parameters():
                parameter.add_(torch.randn_like(parameter) * .003)
        coordinates = torch.tensor([[.17, -.23, .09], [-.31, .12, .28]],
                                   dtype=torch.float64, requires_grad=True)
        time = torch.tensor(.37, dtype=torch.float64, requires_grad=True)
        scale = coordinates.new_tensor([.13, .27, .51])
        actual = pair.prepare().at(time).derivatives(coordinates, scale,
            differential=True, compile_derivatives=False)
        expected = reference_derivatives(pair, coordinates, time, scale)
        for value, truth in zip(actual, expected):
            torch.testing.assert_close(value, truth, rtol=1e-8, atol=1e-10)
        weights = tuple(torch.randn_like(value) for value in actual)
        inputs = (*pair.parameters(), coordinates, time)
        gradients = torch.autograd.grad(actual, inputs, weights, retain_graph=True)
        references = torch.autograd.grad(expected, inputs, weights)
        for value, truth in zip(gradients, references):
            torch.testing.assert_close(value, truth, rtol=1e-8, atol=1e-10)

        # The fixed row factors must survive the actual checkpoint protocol.
        buffer = io.BytesIO()
        torch.save({'model': pair.model_config, 'state': pair.state_dict()}, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        restored = PairedSiren(3, **checkpoint['model']).double()
        restored.load_state_dict(checkpoint['state'], strict=True)
        torch.testing.assert_close(restored.image(coordinates, time),
                                   pair.image(coordinates, time), rtol=0, atol=0)

    def test_default_adam_step_and_learning_rate_match_previous_recipe(self):
        torch.manual_seed(42)
        pair = PairedSiren(3, width=5, layers=2, omega=30.).double()
        reference = copy.deepcopy(pair)
        settings = copy.deepcopy(DEFAULT['training'])
        optimizers = make_optimizers(pair, settings)
        self.assertEqual(len(optimizers), 1)
        old = torch.optim.Adam(reference.parameters(), lr=settings['lr'])
        for model, optimizer in ((pair, optimizers[0]), (reference, old)):
            sum(parameter.square().sum() for parameter in model.parameters()).backward()
            optimizer.step()
        for actual, expected in zip(pair.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for step in [1, 1000, 5000]:
            self.assertEqual(learning_rate_factor(settings, step),
                settings['lr_final_factor'] ** ((step - 1) / settings['steps']))

    def test_first_layer_optimizer_groups_include_raw_frequency_parameters_once(self):
        pair = PairedSiren(3, width=5, layers=2, first_omega_range=[10., 40.],
                           first_omega_mode='parameterized')
        settings = {**DEFAULT['training'], 'first_layer_lr_factor': .25}
        optimizer, = make_optimizers(pair, settings)
        groups = optimizer.param_groups
        ids = [id(parameter) for group in groups for parameter in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(parameter) for parameter in pair.parameters()})
        first = {id(parameter) for model in (pair.source, pair.target)
                 for parameter in model.net[0].parameters()}
        self.assertEqual({id(parameter) for parameter in groups[0]['params']}, first)
        self.assertEqual(groups[0]['lr'], settings['lr'] * .25)

    def test_registration_schedule_reaches_final_weight_and_stays_there(self):
        for name in ['linear', 'cosine']:
            settings = {**DEFAULT['training'], 'registration_weight_schedule':
                        {'name': name, 'start_weight': .1, 'decay_steps': 2500}}
            validate_optimization(settings)
            values = [registration_weight(settings, .01, step)
                      for step in range(1, 5001)]
            self.assertAlmostEqual(values[0], .1)
            self.assertAlmostEqual(values[2499], .01)
            self.assertAlmostEqual(values[-1], .01)
            self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))
        self.assertEqual(registration_weight(DEFAULT['training'], .01, 100), .01)


if __name__ == '__main__':
    unittest.main()
