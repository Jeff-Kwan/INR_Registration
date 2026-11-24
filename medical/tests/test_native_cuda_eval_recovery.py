"""Native FP32 RK2 parity at boundaries and off-grid points."""
import os
import unittest

import torch

from medical.registration.geometry import identity_grid, sample
from medical.registration.recovery import integrate_interval


@unittest.skipUnless(os.environ.get('INR_TEST_CUDA') == '1', 'Opt-in CUDA validation')
class TestNativeCuda(unittest.TestCase):
    def test_midpoint_matches_grid_sample_both_directions(self):
        torch.manual_seed(42)
        shape = (17, 13, 23)
        identity = identity_grid(shape, 'cuda')
        mapping = identity + torch.randn_like(identity)
        # Include a noncontiguous vector field, as returned by some solvers.
        field = torch.randn((3, *shape), device='cuda').movedim(0, -1)
        for dt, substeps in [(0.25, 4), (-0.125, 8)]:
            reference = mapping
            step = dt / substeps
            for _ in range(substeps):
                midpoint = reference + (0.5 * step) * sample(field, reference)
                reference = reference + step * sample(field, midpoint)
            with torch.no_grad():
                actual = integrate_interval(mapping, field, dt, substeps)
            torch.testing.assert_close(actual, reference, rtol=1e-6, atol=3e-6)
