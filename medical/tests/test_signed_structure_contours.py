"""Pure signed-label and slice selection tests; no recovery."""
import unittest
import numpy as np
from scripts.create_structure_contours import signed_targets,target_slice


class TestSignedStructures(unittest.TestCase):
    def test_win_and_loss_are_separate_and_strict(self):
        targets=signed_targets([1,2,3],[.9,.6,.8],[.7,.8,.8])
        self.assertEqual(targets['win']['code'],1);self.assertEqual(targets['loss']['code'],2)
        self.assertIsNone(signed_targets([1,2],[.8,.9],[.7,.8])['loss'])
        self.assertIsNone(signed_targets([1],[.8],[.8])['win'])

    def test_slice_maximizes_target_count_not_middle(self):
        mask=np.zeros((5,6,7),bool);mask[:4,:4,1]=True;mask[0,0,3]=True
        self.assertEqual(target_slice(mask),1)
        mask[:4,:4,5]=True;self.assertEqual(target_slice(mask),1)
        with self.assertRaises(ValueError):target_slice(np.zeros_like(mask))
