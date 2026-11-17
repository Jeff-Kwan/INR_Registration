"""CPU-only temporal indexing and Patch NCC tests; no model inference."""
import unittest
import numpy as np
from scripts.create_acdc_trajectory import bracket,patch_ncc_2d


class TestACDCTrajectory(unittest.TestCase):
    def test_frame_indexing_and_wraparound(self):
        self.assertEqual(bracket(1,14,.25,30),(3,4,.25,4.25))
        self.assertEqual(bracket(1,14,1.,30),(13,13,0.,14.))
        self.assertEqual(bracket(25,5,1.,30),(4,4,0.,35.))

    def test_patch_ncc_is_signed_and_affine_invariant(self):
        rng=np.random.default_rng(42);image=rng.random((20,21)).astype(np.float32)
        self.assertAlmostEqual(patch_ncc_2d(image,image)['mean'],1.)
        self.assertAlmostEqual(patch_ncc_2d(image*2+3,image)['mean'],1.)
        self.assertAlmostEqual(patch_ncc_2d(-image,image)['mean'],-1.)
        self.assertIsNone(patch_ncc_2d(image,np.ones_like(image))['mean'])
