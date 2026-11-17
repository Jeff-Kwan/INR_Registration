"""Corrected MRI recipe and replacement-boundary checks; no fitting."""
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
import torch
from baselines.nir import prepare_array
from baselines.precision import ieee_fp32
from baselines.sources import upstream_imports,native_modules
from baselines.rerun_neural import purge_method_outputs,replacement_plan,ROOT


class RerunCorrections(unittest.TestCase):
    def test_mri_settings_preserve_frozen_intensities(self):
        for ds in ('neurite_oasis','acdc','lpba40','ixi'):
            c=json.loads((ROOT/f'baselines/configs/nir_{ds}.json').read_text())
            self.assertEqual(c['native']['loss_type'],'ncc+discrete_jacobian')
            self.assertEqual(c['native']['gaussian_scale'],3)
            self.assertEqual(c['preprocessing'],'preserve_prepared')
        a=np.array([0.,.8,3.62],dtype=np.float32)
        self.assertIs(prepare_array(a,'preserve_prepared'),a)

    def test_precision_scope_restores_on_exception(self):
        a=torch.backends.cudnn.allow_tf32;b=torch.backends.cuda.matmul.allow_tf32
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with ieee_fp32():
                self.assertFalse(torch.backends.cudnn.allow_tf32)
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                raise RuntimeError('injected')
        self.assertEqual(a,torch.backends.cudnn.allow_tf32)
        self.assertEqual(b,torch.backends.cuda.matmul.allow_tf32)

    def test_recorded_source_spectral_settings_and_ncc_gradient(self):
        with upstream_imports('nir') as path:
            _,modules,efficient,_,_=native_modules('nir')
            import loss_functions
            x=torch.tensor([.1,.2,.3]);torch.testing.assert_close(modules.Sine()(x),torch.sin(30*x))
            encoding=efficient.PosEncodingFourier(in_features=3,gaussian_scale=3,embedding_size=1)
            encoding.bvals.copy_(torch.tensor([[1.,0.,0.]]))
            p=torch.tensor([[.125,0.,0.]])
            expected=torch.tensor([[np.sin(np.pi/4),np.cos(np.pi/4)]],dtype=torch.float32)
            torch.testing.assert_close(encoding(p),expected)
            torch.manual_seed(42);truth=torch.rand(1,13,14,15,1)*.4
            prediction=(truth*.9+.01).requires_grad_(True)
            value=loss_functions.ncc(dict(model_out=2*prediction-1),dict(img=2*truth-1))['ncc']
            self.assertTrue(torch.isfinite(value))
            gradient=torch.autograd.grad(value,prediction)[0];self.assertTrue(torch.isfinite(gradient).all())
            source=(path/'src/training.py').read_text();part=source[source.index('def train_efficient_pairwise('):];part=part[:part.index('\ndef ',5)]
            self.assertIn('GradScaler(enabled=False)',part);self.assertIn('autocast(enabled=False)',part)

    def test_delete_only_explicitly_replaced_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('nir','dnvf','nodeo','ccidir'):
                p=root/'ixi'/name;p.mkdir(parents=True);(p/'checkpoint.pt').write_text('synthetic')
            purge_method_outputs(root,datasets=('ixi',))
            self.assertFalse((root/'ixi/nir').exists());self.assertFalse((root/'ixi/dnvf').exists())
            self.assertTrue((root/'ixi/nodeo/checkpoint.pt').exists());self.assertTrue((root/'ixi/ccidir/checkpoint.pt').exists())
            outside=root/'outside';outside.mkdir();(root/'ixi/nir').symlink_to(outside,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'symlink'):purge_method_outputs(root,datasets=('ixi',))
            self.assertTrue(outside.exists())

    def test_plan_only_has_no_execution_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'absent';plan=replacement_plan(root)
            self.assertEqual(len(plan['jobs']),618)
            self.assertFalse(any(j['missing_inputs'] for j in plan['jobs']))
            self.assertFalse(root.exists());self.assertFalse(plan['execution'])
