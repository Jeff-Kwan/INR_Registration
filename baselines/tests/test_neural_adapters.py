"""Native source/shape and mocked fitting checks; no optimizer steps."""
from contextlib import nullcontext
import json
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from baselines import nodeo, dnvf
from baselines.sources import ROOT, native_modules, upstream_imports, git


def config(name):
    return json.loads((ROOT/f'configs/{name}_neurite_oasis.json').read_text())


class NeuralTests(unittest.TestCase):
    def test_nodeo_patch_fits_all_native_depths(self):
        with upstream_imports('nodeo'):
            _, network, _ = native_modules('nodeo')
            for shape in [(160,192,224),(160,192,160),*[(128,128,z) for z in (19,24,25,27,29,30,32)]]:
                with torch.device('meta'):
                    model=network.BrainNet(shape,'AK',15,1,2,16)
                    result=model(torch.zeros(1,3,*shape))
                self.assertEqual(tuple(result.shape),(1,3,*shape))

    def test_nodeo_original_supported_shape_is_numerically_preserved(self):
        old_threads=torch.get_num_threads();torch.set_num_threads(2)
        try:
            with upstream_imports('nodeo') as root:
                _, network, _ = native_modules('nodeo')
                namespace={}
                exec(compile(git(root,'show','HEAD:Network.py'),'native-Network.py','exec'),namespace)
                shape=(160,192,144)
                torch.manual_seed(42);original=namespace['BrainNet'](shape,'AK',15,1,2,16)
                torch.manual_seed(42);localized=network.BrainNet(shape,'AK',15,1,2,16)
                for a,b in zip(original.parameters(),localized.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
                with torch.no_grad():
                    x=torch.zeros(1,3,*shape)
                    expected=original(x);actual=localized(x)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        finally:torch.set_num_threads(old_threads)

    def test_nodeo_native_call_and_normalized_map_axes(self):
        shape=(5,6,7);grid=np.indices(shape).transpose(1,2,3,0).astype(np.float32)
        normalized=torch.from_numpy((2*grid/(np.array(shape)-1)-1)[...,::-1].copy())[None]
        native=SimpleNamespace(registration=Mock(return_value=(None,normalized,None)))
        case={k:dict(data=np.zeros(shape,np.float32)) for k in ('fixed','moving')}
        points=np.array([[1.,2.,3.]],np.float32)
        def score(name,case,settings,predict,checkpoint,started,**kwargs):
            np.testing.assert_allclose(predict(points),points,atol=1e-6)
            self.assertFalse(checkpoint['parameter_checkpoint'])
            return {}
        with patch.object(nodeo,'preflight'),patch.object(nodeo,'upstream_imports',return_value=nullcontext()), \
             patch.object(nodeo,'execution',return_value=nullcontext(torch.device('cpu'))), \
             patch.object(nodeo,'native_modules',return_value=[native,None,None]), \
             patch.object(nodeo,'finish',side_effect=score):
            nodeo.run_case(case,config('nodeo'),'cuda:0')
        settings=native.registration.call_args.args[0]
        self.assertEqual((settings.epoches,settings.lr,settings.optimizer),(300,.005,'Euler'))

    def test_dnvf_native_entry_capture_and_scratch_cleanup(self):
        shape=(5,6,7);case={k:dict(data=np.zeros(shape,np.float32)) for k in ('moving','fixed')}
        source=SimpleNamespace()
        source.Siren=lambda:torch.nn.Linear(1,1)
        class Integration(torch.nn.Module):
            def __init__(self,*args,**kwargs):super().__init__()
            def forward(self,vector):return vector
        source.VecInt=Integration
        datasets=SimpleNamespace(load_test_dataset=lambda name:None)
        utils=SimpleNamespace(dice_OASIS=lambda *args:None)
        scratches=[]
        def load(path):
            self.assertEqual(Path(path).name,'opt_nf_diff.py')
            def main():
                import sys
                scratch=Path(sys.argv[sys.argv.index('--save_dir')+1]);scratches.append(scratch)
                (scratch/'native.csv').write_text('logging')
                inputs,actual_shape=datasets.load_test_dataset('OASIS')
                self.assertEqual(actual_shape,shape)
                self.assertEqual(len(inputs[0]),4)
                self.assertEqual(int(inputs[0][2].sum()),0)
                source.Siren()
                integrator=source.VecInt(shape)
                for _ in range(config('dnvf')['native']['optimization_epoch']):
                    integrator(torch.zeros(1,3,*shape))
            return dict(main=main)
        def finish(name,case,settings,predict,checkpoint,started,**kwargs):
            points=np.array([[1.,2.,3.]],np.float32)
            np.testing.assert_array_equal(predict(points),points)
            self.assertIn('state',checkpoint)
            return {}
        for failed in (False,True):
            with patch.object(dnvf,'preflight'),patch.object(dnvf,'upstream_imports',return_value=nullcontext(ROOT/'upstream/dnvf')), \
                 patch.object(dnvf,'execution',return_value=nullcontext()), \
                 patch.object(dnvf,'native_modules',return_value=[source,datasets,utils]), \
                 patch.object(dnvf.runpy,'run_path',side_effect=load), \
                 patch.object(dnvf,'finish',side_effect=RuntimeError('evaluation') if failed else finish):
                if failed:
                    with self.assertRaises(RuntimeError):dnvf.run_case(case,config('dnvf'),'cuda:0')
                else:dnvf.run_case(case,config('dnvf'),'cuda:0')
        self.assertTrue(all(not p.exists() for p in scratches))
