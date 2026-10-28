"""Independent higher-order AD references for analytical differential LK."""
import copy
import unittest
from unittest.mock import patch

import torch

from cameraman.core.inr import Siren
from cameraman.core.lk_loss import (analytical_siren_derivatives, lk_temporal_loss,
                     minimized_transport_energy, minimized_differential_transport_energy)

from medical.models.siren3d import Siren3D


from cameraman.core.derivative_reference import autograd_reference


class DifferentialLKTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)

    def test_hessians_mixed_derivatives_and_parameter_gradients_against_ad(self):
        for constructor, d in ((Siren, 2), (Siren3D, 3)):
            for layers in (1, 3):
                a = constructor(width=6, layers=layers, omega=13.).double()
                b = copy.deepcopy(a)
                with torch.no_grad():
                    for p in b.parameters():
                        p.add_(torch.randn_like(p) * .005)
                x = torch.rand(2, 7, d, dtype=torch.float64) * 2 - 1
                t = x.new_tensor(.37)
                scale = [2 / n for n in (17, 23, 31)[:d]]
                actual = analytical_siren_derivatives(a, b, x, t, scale, differential=True)
                reference = autograd_reference(a, b, x, t, scale)
                for value, expected in zip(actual, reference):
                    torch.testing.assert_close(value, expected, atol=1e-12, rtol=1e-10)
                torch.testing.assert_close(actual[2], actual[2].transpose(-1, -2), atol=1e-12, rtol=1e-10)
                loss0 = minimized_differential_transport_energy(*actual, 1e-6, 2.3)[0]
                loss1 = minimized_differential_transport_energy(*reference, 1e-6, 2.3)[0]
                torch.testing.assert_close(loss0, loss1, atol=1e-12, rtol=1e-10)
                parameters = tuple(a.parameters()) + tuple(b.parameters())
                grads0, grads1 = [torch.autograd.grad(loss, parameters) for loss in (loss0, loss1)]
                for value, expected in zip(grads0, grads1):
                    self.assertTrue(torch.isfinite(value).all())
                    torch.testing.assert_close(value, expected, atol=1e-11, rtol=1e-9)
                for grads in (grads0[:len(parameters)//2], grads0[len(parameters)//2:]):
                    self.assertGreater(sum(float(v.abs().sum()) for v in grads), 0)

    def test_exact_translation_in_anisotropic_2d_and_3d_coordinates(self):
        for constructor, d in ((Siren, 2), (Siren3D, 3)):
            a = constructor(width=d, layers=1, omega=1.).double()
            frequency = torch.tensor([2., 3., 5.][:d], dtype=torch.float64)
            scale = torch.tensor([2/16, 2/22, 2/30][:d], dtype=torch.float64)
            velocity = torch.tensor([1.7, -.8, .4][:d], dtype=torch.float64)
            with torch.no_grad():
                a.net[0].linear.weight.copy_(torch.diag(frequency))
                a.net[0].linear.bias.zero_()
                a.net[-1].weight.fill_(1.)
                a.net[-1].bias.zero_()
            b = copy.deepcopy(a)
            with torch.no_grad():
                b.net[0].linear.bias.copy_(-frequency * scale * velocity)
            x = torch.rand(3, 32, d, dtype=torch.float64) * 2 - 1
            t = x.new_tensor(.4)
            g, q, hessian, mixed = analytical_siren_derivatives(a, b, x, t, scale, differential=True)
            phase = frequency * (x - t * scale * velocity)
            expected_h = torch.diag_embed(-frequency.square() * scale.square() * phase.sin())
            torch.testing.assert_close(hessian, expected_h, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(q, -(g * velocity).sum(-1), atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(mixed, -(hessian @ velocity[..., None]).squeeze(-1), atol=1e-12, rtol=1e-12)
            loss, v = minimized_differential_transport_energy(g, q, hessian, mixed, 1e-12, 3.)
            torch.testing.assert_close(v, velocity.expand_as(v), atol=1e-7, rtol=1e-7)
            self.assertLess(float(loss.detach()), 1e-10)

    def test_alpha_zero_is_exact_and_skips_hessians(self):
        a, b = Siren(6, 2).double(), Siren(6, 2).double()
        x = torch.rand(2, 9, 2, dtype=torch.float64)
        t = x.new_tensor(.3)
        g, q = analytical_siren_derivatives(a, b, x, t, [2/16, 2/22])
        old = minimized_transport_energy(g, q, 1e-6)
        new = minimized_differential_transport_energy(g, q, None, None, 1e-6, 0)
        for actual, expected in zip(new, old):
            self.assertTrue(torch.equal(actual, expected))
        with patch('cameraman.core.lk_loss.analytical_siren_derivatives', wraps=analytical_siren_derivatives) as spy:
            loss = lk_temporal_loss(a, b, x, t, (23, 17), gradient_weight=0)
            self.assertNotIn('differential', spy.call_args.kwargs)
        self.assertTrue(torch.equal(loss, old[0]))
        parameters = tuple(a.parameters()) + tuple(b.parameters())
        g0 = torch.autograd.grad(old[0], parameters, retain_graph=True)
        g1 = torch.autograd.grad(new[0], parameters)
        for value, expected in zip(g0, g1):
            self.assertTrue(torch.equal(value, expected))

    def test_normal_equations_and_gradient_check(self):
        g = torch.randn(1, 4, 3, dtype=torch.float64, requires_grad=True)
        q = torch.randn(1, 4, dtype=torch.float64, requires_grad=True)
        h = torch.randn(1, 4, 3, 3, dtype=torch.float64, requires_grad=True)
        mixed = torch.randn(1, 4, 3, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda *v:
            minimized_differential_transport_energy(*v, .02, 1.7)[0], (g, q, h, mixed)))
        loss, velocity = minimized_differential_transport_energy(g, q, h, mixed, .02, 1.7)
        # Independent stacked least-squares reference (sum over gradient channels).
        design = torch.cat((g, h.flatten(1, 2) * 1.7**.5), dim=1)
        rhs = -torch.cat((q, mixed.flatten(1) * 1.7**.5), dim=1)
        matrix = design.transpose(-1, -2) @ design / 4 + .02 * torch.eye(3, dtype=g.dtype)
        wanted = torch.linalg.solve(matrix, (design.transpose(-1, -2) @ rhs[..., None]) / 4).squeeze(-1)
        torch.testing.assert_close(velocity, wanted, atol=1e-12, rtol=1e-12)
        self.assertTrue(torch.isfinite(loss))

    def test_flat_and_rank_deficient_stability_and_validation(self):
        for d in (2, 3):
            for flat in (True, False):
                g = torch.zeros(2, 7, d, requires_grad=True)
                h = torch.zeros(2, 7, d, d, requires_grad=True)
                with torch.no_grad():
                    if not flat:
                        g[..., 0] = .1
                        h[..., 0, 0] = .01
                q = torch.ones(2, 7, requires_grad=True)
                mixed = torch.ones(2, 7, d, requires_grad=True)
                loss, v = minimized_differential_transport_energy(g, q, h, mixed, 1e-6, 100.)
                gradients = torch.autograd.grad(loss, (g, q, h, mixed))
                self.assertTrue(all(torch.isfinite(value).all() for value in (*gradients, loss, v)))
        for alpha in (-1, float('nan')):
            with self.assertRaises(ValueError):
                minimized_differential_transport_energy(g, q, h, mixed, 1e-6, alpha)


if __name__ == '__main__':
    unittest.main()
