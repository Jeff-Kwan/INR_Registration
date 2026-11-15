"""Parity checks for in-memory CUDA Learn2Reg scoring against the pinned metrics."""
import os
from unittest import TestCase, skipUnless
from unittest.mock import patch

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
import torch

from medical import abdomen_evaluation as evaluation


@skipUnless(os.environ.get('INR_TEST_CUDA') == '1' and torch.cuda.is_available(),
            'CUDA scorer parity is opt-in')
class TestLearn2RegCuda(TestCase):
    def setUp(self):
        self.shape = (13, 11, 7)
        self.identity = np.stack(np.indices(self.shape), axis=-1).astype(np.float32)
        rng = np.random.default_rng(9)
        self.moving = rng.integers(0, 14, self.shape, dtype=np.uint8)
        self.fixed = rng.integers(0, 13, self.shape, dtype=np.uint8)
        self.case = {key: {'path': f'/synthetic/imagesTr/{key}.nii.gz'}
                     for key in ('fixed', 'moving')}

    def scores(self, mapping, fixed, moving):
        # Labels live entirely in memory. Neither image data nor maps are written.
        images = [nib.Nifti1Image(array, np.eye(4)) for array in (fixed, moving)]
        with patch.object(evaluation.nib, 'load', side_effect=images):
            return evaluation.evaluate(self.case, mapping)

    def test_signed_jacobian_empty_and_missing_labels(self):
        mapping = self.identity.copy()
        # Seven of nine interior X planes fold. For 147/189 voxels, CUDA
        # mean-by-reciprocal differs from the published fraction by one ULP.
        mapping[..., 0] = .2 * (self.identity[..., 0] - 8.25)**2
        mapping[..., 1] += .5
        mapping[0, 0, 0] = [np.nextafter(np.float32(.5), np.float32(0)), 0., 0.]
        official = evaluation.official_functions()
        labels = list(range(1, 14))
        for pins in (mapping, mapping + 1000):
            # The real native map has XYZ axes in a strided ZYX allocation.
            gpu = torch.tensor(pins.transpose(2, 1, 0, 3).copy(), device='cuda').permute(2, 1, 0, 3)
            result = self.scores(gpu, self.fixed, self.moving)
            pulled = map_coordinates(self.moving, pins.transpose(3, 0, 1, 2), order=0)
            for key, metric in [('DSC', official.compute_dice), ('HD95', official.compute_hd95)]:
                mean, detailed = metric(self.fixed, self.moving, pulled, labels)
                self.assertEqual(result[key]['detailed'], [evaluation.json_metric(x) for x in detailed])
                self.assertEqual(result[key]['mean'], evaluation.json_metric(mean))
            disp = pins - self.identity.astype(np.float64)
            jac = official.jacobian_determinant(disp[None].transpose(0, 4, 1, 2, 3))
            self.assertEqual(result['folding_fraction'], (jac <= 0).mean())
            self.assertAlmostEqual(result['LogJacDetStd']['mean'],
                                   np.log((jac + 3).clip(1e-9, 1e9)).std(), places=12)
            self.assertEqual(result['num_foldings']['mean'], 0.)

    def test_floating_label_fallback(self):
        fixed = self.fixed.astype(np.float32)
        fixed[fixed == 1] += .25
        cpu = self.scores(self.identity, fixed, self.moving)
        gpu = self.scores(torch.as_tensor(self.identity, device='cuda'), fixed, self.moving)
        self.assertEqual(cpu, gpu)
