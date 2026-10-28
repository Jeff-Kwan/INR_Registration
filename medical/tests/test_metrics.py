"""MONAI Dice/HD95 conventions, physical units and missing-label regression checks."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from monai.metrics import compute_dice, compute_hausdorff_distance
from scipy.ndimage import binary_erosion, distance_transform_edt

from medical.registration.io import label_metrics, write_json


class MonaiLabelMetricTests(unittest.TestCase):
    def test_matches_monai_multiclass_macro_average_and_spacing(self):
        fixed=np.zeros((12,13,14),np.int16)
        fixed[1:5,2:7,2:8]=1
        fixed[7:10,8:11,8:12]=2
        warped=np.roll(fixed,1,axis=0)
        spacing=(2.,1.5,.7)
        actual,per_label=label_metrics(fixed,warped,spacing)
        a=torch.from_numpy(np.stack([fixed==i for i in (1,2)]))[None]
        b=torch.from_numpy(np.stack([warped==i for i in (1,2)]))[None]
        dice=compute_dice(b,a,include_background=True,ignore_empty=False).numpy().ravel()
        hd=compute_hausdorff_distance(b,a,include_background=True,percentile=95,
                                     directed=False,spacing=spacing).numpy().ravel()
        self.assertEqual(actual['label_count'],2)
        self.assertNotIn('0',per_label)
        for i in range(2):
            self.assertAlmostEqual(per_label[str(i+1)]['dice'],float(dice[i]))
            self.assertAlmostEqual(per_label[str(i+1)]['hd95_mm'],float(hd[i]))
        self.assertAlmostEqual(actual['dice'],float(np.mean(dice,dtype=np.float64)))
        self.assertAlmostEqual(actual['hd95_mm'],float(np.mean(hd,dtype=np.float64)))
        self.assertEqual(actual['metric_definition']['implementation'],'MONAI')

    def test_directional_hd95_does_not_pool_away_one_sided_error(self):
        a=np.zeros((24,24,24),bool);a[2:10,2:10,2:10]=True
        b=a.copy();b[18:21,18:21,18:21]=True
        sa=a & ~binary_erosion(a,border_value=0)
        sb=b & ~binary_erosion(b,border_value=0)
        ab=distance_transform_edt(~sb)[sa]
        ba=distance_transform_edt(~sa)[sb]
        self.assertEqual(float(np.percentile(np.concatenate((ab,ba)),95)),0.)
        expected=max(np.percentile(ab,95),np.percentile(ba,95))
        actual,_=label_metrics(a,b,(1.,1.,1.))
        reverse,_=label_metrics(b,a,(1.,1.,1.))
        self.assertGreater(actual['hd95_mm'],0.)
        self.assertAlmostEqual(actual['hd95_mm'],float(expected),places=5)
        self.assertEqual(actual['hd95_mm'],reverse['hd95_mm'])

    def test_missing_and_spurious_labels_are_not_silently_ignored(self):
        a=np.zeros((5,6,7),np.int16);a[1,2,3]=1
        b=a.copy();b[3,4,5]=2
        metrics,per_label=label_metrics(a,b,(1.,1.,1.))
        self.assertEqual(metrics['dice'],.5)
        self.assertEqual(per_label['2']['dice'],0.)
        self.assertIsNone(metrics['hd95_mm'])
        self.assertTrue(metrics['hd95_infinite'])
        self.assertTrue(per_label['2']['hd95_infinite'])
        self.assertEqual(metrics['missing_labels'],[2])
        self.assertEqual(metrics['hd95_finite_labels_mean_mm'],0.)
        reverse,_=label_metrics(b,a,(1.,1.,1.))
        self.assertEqual(reverse['dice'],.5)
        empty,_=label_metrics(np.zeros_like(a),np.zeros_like(a),(1.,1.,1.))
        self.assertIsNone(empty['dice'])
        self.assertIsNone(empty['hd95_mm'])
        self.assertEqual(empty['label_count'],0)

    def test_sweep_rejects_unversioned_historical_metrics(self):
        from medical.experiments.hyperparameter_sweep import collect_metrics
        a=np.zeros((4,5,6),np.int16);a[1:3,1:4,1:5]=1
        labels,_=label_metrics(a,a,(1.,1.,1.))
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);subject=root/'S02';subject.mkdir()
            write_json(subject/'status.json',{'complete':True})
            write_json(subject/'metrics.json',{'labels':labels,'endpoint_quality_pass':True})
            self.assertEqual(collect_metrics(root,['S02'])[0]['dice'],1.)
            labels.pop('metric_definition')
            write_json(subject/'metrics.json',{'labels':labels,'endpoint_quality_pass':True})
            with self.assertRaisesRegex(ValueError,'metric definition'):
                collect_metrics(root,['S02'])

    def test_dice_only_still_uses_monai_and_skips_hd(self):
        a=np.zeros((4,5,6),np.int16);a[1:3,1:4,1:5]=1
        with patch('medical.registration.io.compute_hausdorff_distance',side_effect=AssertionError('HD disabled')):
            scores,_=label_metrics(a,a,(1.,2.,3.),compute_hd95=False)
        self.assertEqual(scores['dice'],1.)
        self.assertIsNone(scores['hd95_mm'])
        self.assertFalse(scores['metric_definition']['hd95_enabled'])
        for spacing in ((0,1,1),(-1,1,1),(1,np.nan,1),(1,1)):
            with self.assertRaises(ValueError): label_metrics(a,a,spacing)
        with self.assertRaises(ValueError): label_metrics(a,a[:2],(1,1,1))


if __name__=='__main__':
    unittest.main()
