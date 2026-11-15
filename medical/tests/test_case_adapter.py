"""Adapter tests with mocked fit/recovery; no optimizer or research run."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import nibabel as nib
import torch
from medical.configuration import load_config
from medical.paired_method import run_case
from medical.registration.geometry import identity_grid
from medical.registration.recovery import RecoveryResult


class TestAdapter(unittest.TestCase):
    def test_identity_map_scores_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td); pts=np.tile([2,2,2],(300,1))
            for name in ('T00','T50'):
                np.savetxt(d/f'{name}_landmarks_voxel.csv',pts,delimiter=',',header='x,y,z',comments='')
            data=np.arange(125,dtype=np.float32).reshape(5,5,5)
            case={'fixed':{'data':data,'affine':np.eye(4),'path':str(d/'T00.nii.gz')},
                  'moving':{'data':data,'affine':np.eye(4),'path':str(d/'T50.nii.gz')}}
            nib.save(nib.Nifti1Image(np.ones((5,5,5),dtype=np.uint8),np.eye(4)),d/'T00_lung_mask.nii.gz')
            nib.save(nib.Nifti1Image(np.ones((5,5,5),dtype=np.uint8),np.eye(4)),d/'T50_lung_mask.nii.gz')
            pair=torch.nn.Linear(1,1); pair.model_config={'width':1}
            field=identity_grid((5,5,5),'cpu')
            with patch('medical.paired_method.fit_pair',return_value=pair), patch('medical.paired_method.recover',return_value=RecoveryResult(field,field,[])):
                result=run_case(case,load_config(),'cpu')
            self.assertEqual(result['metrics']['idir_rounded_tre_mm']['mean_mm'],0)
            self.assertEqual(result['metrics']['folding_fraction'],0)
            self.assertIn('state',result['checkpoint'])
            self.assertEqual(result['metrics']['reverse_T50_to_T00']['continuous_tre_mm']['mean_mm'],0)
