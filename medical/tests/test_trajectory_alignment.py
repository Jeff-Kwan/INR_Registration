"""Synthetic temporal DP and metric parity; no model inference."""
import unittest
import numpy as np
from scripts.align_acdc_trajectory import monotone_alignment,score_alignment,matrix


class TestTrajectoryAlignment(unittest.TestCase):
    def test_linear_similarity_prefers_diagonal(self):
        scores=np.ones((53,14))
        result=monotone_alignment(scores)
        self.assertEqual(result['indices'],list(range(0,53,4)))
        self.assertEqual(score_alignment(scores,result)['aligned']['mean'],1.)

    def test_known_ordered_warp_is_recovered_and_bounded(self):
        target=np.array([0,1,3,6,10,14,18,23,29,35,41,46,50,52])
        scores=1-((np.arange(53)[:,None]-target[None,:])/20)**2
        result=monotone_alignment(scores,speed_penalty=0,smooth_penalty=0)
        self.assertEqual(result['indices'],target.tolist())
        steps=np.diff(result['indices']);self.assertTrue((steps>=1).all() and (steps<=16).all())
        result=monotone_alignment(scores)
        summary=score_alignment(scores,result)
        self.assertGreaterEqual(summary['aligned']['mean'],summary['unaligned']['mean'])
        self.assertGreaterEqual(summary['independent_best']['mean'],summary['aligned']['mean'])
        with self.assertRaises(ValueError):monotone_alignment(scores,max_step=1)

    def test_matrix_matches_existing_patch_ncc(self):
        import torch
        from medical.sirenpath_reconstruction import patch_ncc
        rng=np.random.default_rng(42);a=rng.random((11,12,13)).astype(np.float32);b=rng.random(a.shape).astype(np.float32)
        scores=matrix([a],[b],'cpu')
        expected=patch_ncc(torch.from_numpy(a.transpose(2,1,0).copy()),torch.from_numpy(b.transpose(2,1,0).copy()))['mean']
        self.assertAlmostEqual(scores[0,0],expected,places=12)
