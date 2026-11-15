"""Native FP32 RK2 and DIR-Lab metric parity at boundaries and off-grid points."""
import os
import unittest

import numpy as np
import torch

from medical.dirlab_evaluation import score_landmarks
from medical.registration.geometry import identity_grid, sample
from medical.registration.recovery import integrate_interval


@unittest.skipUnless(os.environ.get('INR_TEST_CUDA') == '1', 'Opt-in CUDA validation')
class TestNativeCuda(unittest.TestCase):
    def test_midpoint_matches_grid_sample_both_directions(self):
        from medical.registration.dirlab_cuda import folding_fractions
        from medical.registration.metrics import jacobian_determinant

        torch.manual_seed(42)
        shape = (17, 13, 23)
        identity = identity_grid(shape, 'cuda')
        mapping = identity + torch.randn_like(identity)
        # Include a noncontiguous vector field, as returned by some solvers.
        field = torch.randn((3, *shape), device='cuda').movedim(0, -1)
        mask = torch.rand(shape, device='cuda') > 0.4
        for dt, substeps in [(0.25, 4), (-0.125, 8)]:
            reference = mapping
            step = dt / substeps
            for _ in range(substeps):
                midpoint = reference + (0.5 * step) * sample(field, reference)
                reference = reference + step * sample(field, midpoint)
            with torch.no_grad():
                actual = integrate_interval(mapping, field, dt, substeps)
            torch.testing.assert_close(actual, reference, rtol=1e-6, atol=3e-6)
            folded = jacobian_determinant((reference - identity).cpu().numpy()) <= 0
            self.assertEqual(folding_fractions(reference, mask),
                             (folded.mean(), folded[mask.cpu().numpy()].mean()))

    def test_landmark_half_voxels_and_oblique_affine(self):
        rng = np.random.default_rng(42)
        fixed, moving = rng.integers(0, 21, size=(2, 300, 3))
        predicted = moving + rng.choice([-1.5, -.5, .5, 1.5], size=(300, 3))
        affine = np.array([[0, -1.3, 0, 10], [.8, 0, 0, 8],
                           [0, 0, 2.5, -4], [0, 0, 0, 1]])
        def score(p):
            return score_landmarks(fixed, moving, p, affine, grid_shape=(21, 21, 21))
        cpu, cuda = score(predicted), score(torch.as_tensor(predicted, device='cuda'))
        self.assertEqual(set(cpu), set(cuda))
        for key in cpu:
            if isinstance(cpu[key], dict):
                for metric in cpu[key]:
                    self.assertAlmostEqual(cpu[key][metric], cuda[key][metric], places=12)
            else:
                np.testing.assert_allclose(cpu[key], cuda[key], atol=1e-12, rtol=1e-12)
