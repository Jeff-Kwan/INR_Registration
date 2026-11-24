"""Recipe and zero-weight gradient checks; no optimizer steps or recovery."""
import unittest
import torch
from medical.configuration import load_config, validate_config
from medical.models.siren import PairedSiren
from medical.registration.objective import endpoint_loss, TrainingBatch, endpoint_parameter_penalty
from medical.registration.optimization import learning_rate_factor, registration_weight


class TestCanonicalRecipe(unittest.TestCase):
    def test_user_recipe_and_schedule_endpoints(self):
        config = load_config()
        self.assertEqual(config['model'], dict(width=512, layers=4, omega=30., output_init_scale=30.))
        settings, objective = config['training'], config['objective']
        self.assertEqual((settings['steps'], settings['batch_size']), (5000, 32768))
        self.assertEqual((settings['integrator'], settings['temporal_sampling']), ('rk2', 'sobol'))
        self.assertEqual(objective['registration']['particles'], 65536)
        self.assertEqual(objective['registration']['sigma_cells'], .5)
        self.assertEqual(objective['ncc']['moment_weight'], .005)
        self.assertEqual(objective['parameter_l2'], .01)
        self.assertAlmostEqual(registration_weight(settings, .01, 1), .03)
        self.assertAlmostEqual(registration_weight(settings, .01, 5000), .01)
        self.assertAlmostEqual(learning_rate_factor(settings, 1), 1.)
        self.assertAlmostEqual(learning_rate_factor(settings, 5000), .2)
        factors = [learning_rate_factor(settings, step) for step in range(1, 5001)]
        self.assertTrue(all(a >= b for a, b in zip(factors, factors[1:])))
        self.assertEqual((config['recovery']['intervals'], config['recovery']['substeps'],
                          config['runtime']['recovery_max_axis']), (8, 8, 256))

    def test_every_component_accepts_zero_and_registration_stays_off(self):
        config = load_config()
        objective = config['objective']
        objective.update(weight=0., endpoint_weight=0., brightness_weight=0., parameter_l2=0., c=0.)
        objective['ncc']['moment_weight'] = 0.
        objective['registration']['weight'] = 0.
        config['training']['optimizer']['weight_decay'] = 0.
        config['recovery']['smoothness'] = 0.
        validate_config(config)
        for step in (1, 2500, 5000):
            self.assertEqual(registration_weight(config['training'], 0., step), 0.)

    def test_zero_endpoint_keeps_endpoint_parameter_penalty_independent(self):
        torch.manual_seed(42)
        pair = PairedSiren(width=4, layers=1).double()
        with torch.no_grad():
            pair.target.net[-1].bias.add_(.2)
        x = torch.randn(23, 3, dtype=torch.float64)
        batch = TrainingBatch(x, torch.rand(23, dtype=torch.float64), torch.rand(23, dtype=torch.float64))
        objective = load_config()['objective']
        objective['endpoint_weight'] = 0.
        objective['ncc']['moment_weight'] = 0.
        actual = endpoint_loss(pair, batch, objective)['total']
        endpoints = pair.prepare(temporal=False)
        expected = .01 * endpoint_parameter_penalty(endpoints.source, endpoints.target)
        torch.testing.assert_close(actual, expected)
        a = torch.autograd.grad(actual, tuple(pair.parameters()), retain_graph=True)
        b = torch.autograd.grad(expected, tuple(pair.parameters()))
        for value, truth in zip(a, b):
            torch.testing.assert_close(value, truth)
        objective['parameter_l2'] = 0.
        loss = endpoint_loss(pair, batch, objective)['total']
        self.assertEqual(float(loss.detach()), 0.)
        for gradient in torch.autograd.grad(loss, tuple(pair.parameters())):
            self.assertEqual(float(gradient.abs().sum()), 0.)

    def test_moments_can_be_disabled_without_disabling_ncc(self):
        pair = PairedSiren(width=4, layers=1).double()
        x = torch.randn(23, 3, dtype=torch.float64)
        batch = TrainingBatch(x, torch.rand(23, dtype=torch.float64), torch.rand(23, dtype=torch.float64))
        objective = load_config()['objective']
        objective.update(parameter_l2=0.)
        objective['ncc']['moment_weight'] = 0.
        terms = endpoint_loss(pair, batch, objective)
        self.assertEqual(float(terms['weighted_moments'].detach()), 0.)
        torch.testing.assert_close(terms['total'], terms['endpoint_ncc_loss'])

    def test_ncc_can_be_disabled_without_disabling_moments(self):
        pair = PairedSiren(width=4, layers=1).double()
        x = torch.randn(23, 3, dtype=torch.float64)
        batch = TrainingBatch(x, torch.rand(23, dtype=torch.float64), torch.rand(23, dtype=torch.float64))
        objective = load_config()['objective']
        objective.update(endpoint_weight=0., parameter_l2=0.)
        terms = endpoint_loss(pair, batch, objective)
        torch.testing.assert_close(terms['total'], terms['weighted_moments'])
        self.assertGreater(float(terms['weighted_moments'].detach()), 0.)

    def test_lk_terms_can_be_disabled_independently(self):
        from medical.registration.symmetric_registration import shared_pair_velocity, shared_pair_energy
        from medical.registration.captured_objective import checked_velocity
        torch.manual_seed(42)
        g = torch.randn(7, 4, 3, dtype=torch.float64)
        q = torch.randn(7, 4, dtype=torch.float64)
        h = torch.randn(7, 4, 3, 3, dtype=torch.float64)
        m = torch.randn(7, 4, 3, dtype=torch.float64)
        for brightness, differential in ((0., 4.), (1., 0.), (0., 0.)):
            a = shared_pair_velocity(g, q, h, m, differential, 1e-6, brightness_weight=brightness)
            b = checked_velocity(g, q, h, m, differential, 1e-6, brightness_weight=brightness)
            torch.testing.assert_close(a, b)
            # The disabled data term must have no influence on the solve.
            modified = shared_pair_velocity(g if brightness else g*100, q if brightness else q*100,
                                           h if differential else h*100, m if differential else m*100,
                                           differential, 1e-6, brightness_weight=brightness)
            torch.testing.assert_close(a, modified)
            if not brightness and not differential:
                self.assertEqual(float(a.abs().sum()), 0.)
                self.assertEqual(float(shared_pair_energy(g, q, h, m, a, 0., 1e-6, brightness_weight=0.).abs().sum()), 0.)

    def test_retired_interfaces_are_rejected(self):
        for key in ('working_grid', 'spatial_units'):
            config = load_config(); config[key] = {}
            with self.assertRaises(ValueError): validate_config(config)
        for section, key, value in [('training', 'integrator', 'euler4'),
                                     ('objective', 'endpoint', 'mse'),
                                     ('training', 'optimizer', {'name': 'paired_adamw', 'weight_decay': .01})]:
            config = load_config(); config[section][key] = value
            with self.assertRaises(ValueError): validate_config(config)
        config = load_config(); config['objective']['registration']['particle_velocity'] = {}
        with self.assertRaises(ValueError): validate_config(config)
