"""Manual sine chain rule versus autograd/JVP, including training gradients."""
import copy
import unittest

import torch

from cameraman.core.inr import Siren
from cameraman.core.lk_loss import (analytical_siren_derivatives, exact_image_derivatives,
                     minimized_transport_energy, lk_temporal_loss)

from medical.models.siren3d import Siren3D


class AnalyticalSirenTests(unittest.TestCase):
    def test_values_losses_and_every_endpoint_parameter_gradient_in_2d_and_3d(self):
        torch.manual_seed(23)
        for constructor, d in ((Siren, 2), (Siren3D, 3)):
            for layers in (1, 3, 4):
                a = constructor(width=8, layers=layers, omega=17).double()
                b = copy.deepcopy(a)
                with torch.no_grad():
                    for p in b.parameters():
                        p.add_(torch.randn_like(p) * 0.005)
                x = torch.rand(3, 19, d, dtype=torch.float64) * 2 - 1
                scale = [2 / n for n in (17, 23, 31)[:d]]
                t = x.new_tensor(.39)
                g0, q0 = exact_image_derivatives(a, b, x, t, scale)
                g1, q1 = analytical_siren_derivatives(a, b, x, t, scale)
                torch.testing.assert_close(g1, g0, atol=1e-12, rtol=1e-10)
                torch.testing.assert_close(q1, q0, atol=1e-12, rtol=1e-10)
                spatial, none = analytical_siren_derivatives(a, b, x, t, scale, temporal=False)
                self.assertIsNone(none)
                torch.testing.assert_close(spatial, g0, atol=1e-12, rtol=1e-10)
                l0 = minimized_transport_energy(g0, q0, 1e-6)[0]
                l1 = minimized_transport_energy(g1, q1, 1e-6)[0]
                torch.testing.assert_close(l1, l0, atol=1e-12, rtol=1e-10)
                parameters = tuple(a.parameters()) + tuple(b.parameters())
                grads0 = torch.autograd.grad(l0, parameters)
                grads1 = torch.autograd.grad(l1, parameters)
                for actual, expected in zip(grads1, grads0):
                    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-9)

    def test_rejects_non_sine_architecture(self):
        a = Siren(8, 2)
        a.net[0].activation = torch.nn.GELU()
        with self.assertRaises(ValueError):
            analytical_siren_derivatives(a, copy.deepcopy(a), torch.rand(2, 9, 2), .3, [1, 1])

    def test_training_loss_defaults_to_analytical_and_retains_autograd_switch(self):
        torch.manual_seed(4)
        a, b = Siren(8, 2).double(), Siren(8, 2).double()
        x = torch.rand(2, 9, 2, dtype=torch.float64)
        t = x.new_tensor(.4)
        analytical = lk_temporal_loss(a, b, x, t, (17, 23))
        reference = lk_temporal_loss(a, b, x, t, (17, 23), derivatives='autograd')
        torch.testing.assert_close(analytical, reference, rtol=1e-10, atol=1e-12)
        with self.assertRaises(ValueError):
            lk_temporal_loss(a, b, x, t, (17, 23), derivatives='invalid')


if __name__ == '__main__':
    unittest.main()
