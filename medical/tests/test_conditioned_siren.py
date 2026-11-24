"""Check analytical parameter folding against explicit time conditioning."""
import unittest

import torch
import torch.nn.functional as F

from medical.models.conditioned_siren import ConditionedSiren
from medical.registration.derivatives import _layer_curvature_chain


def direct_field(model, voxels, time, scale):
    coords = voxels * scale - 1.
    tau = 2. * time - 1.
    layers = model.spatial.net
    if model.architecture == 'xyzt':
        weight = torch.cat((layers[0].linear.weight, model.slopes[:, None]), dim=1)
        coords = torch.cat((coords, tau.expand(coords.shape[0], 1)), dim=1)
        values = (layers[0].omega * F.linear(coords, weight, layers[0].linear.bias)).sin()
        for layer in layers[1:-1]:
            values = layer(values)
    else:
        values = coords
        for layer, (scale_slope, shift_slope) in zip(layers[:-1], model.slopes):
            values = (1. + tau * scale_slope) * layer(values) + tau * shift_slope
    return layers[-1](values).flatten()


class TestConditionedSiren(unittest.TestCase):
    def test_values_derivatives_and_training_gradients(self):
        torch.manual_seed(42)
        scale = torch.tensor([0.13, 0.27, 0.51], dtype=torch.float64)
        for architecture in ('xyzt', 'film'):
            with self.subTest(architecture=architecture):
                model = ConditionedSiren(architecture, width=7, layers=3, omega=30.,
                                         output_init_scale=30.).double()
                voxels = torch.rand(6, 3, dtype=torch.float64).requires_grad_()
                time = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
                coordinates = voxels * scale - 1.
                prepared = model.prepare().at(time)
                torch.testing.assert_close(prepared.image(coordinates), model.spatial(coordinates))
                with torch.no_grad():
                    model.slopes.normal_(std=0.02)
                prepared = model.prepare().at(time)
                expected = direct_field(model, voxels, time, scale)
                torch.testing.assert_close(prepared.image(coordinates).flatten(), expected)
                g = torch.autograd.grad(expected.sum(), voxels, create_graph=True)[0]
                q = torch.stack([torch.autograd.grad(value, time, create_graph=True)[0]
                                 for value in expected])
                h = torch.stack([torch.autograd.grad(g[:, axis].sum(), voxels, create_graph=True)[0]
                                 for axis in range(3)], dim=-2)
                m = torch.autograd.grad(q.sum(), voxels, create_graph=True)[0]
                derivatives = _layer_curvature_chain(prepared.parameters, prepared.direction,
                                                      prepared.omegas, coordinates, scale)
                for actual, reference in zip(derivatives, (g, q, h, m)):
                    torch.testing.assert_close(actual, reference, atol=1e-8, rtol=1e-8)
                for compiled, full in zip(prepared.derivatives(coordinates, scale), derivatives):
                    torch.testing.assert_close(compiled, full, atol=1e-8, rtol=1e-8)
                probes = tuple(torch.randn_like(value) for value in derivatives)
                params = tuple(model.parameters())
                a = torch.autograd.grad(derivatives, params, probes, allow_unused=True)
                b = torch.autograd.grad((g, q, h, m), params, probes, allow_unused=True)
                for parameter, actual, reference in zip(params, a, b):
                    torch.testing.assert_close(
                        torch.zeros_like(parameter) if actual is None else actual,
                        torch.zeros_like(parameter) if reference is None else reference,
                        atol=1e-7, rtol=1e-8)
                self.assertTrue(a[0].abs().max() > 0)


if __name__ == '__main__':
    unittest.main()
