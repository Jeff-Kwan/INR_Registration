"""Source-math parity on synthetic inputs. No optimizer, fit or real-data run."""
from contextlib import nullcontext,redirect_stdout
import importlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np
from scipy.ndimage import map_coordinates
import torch

from baselines import ccidir,nir
from baselines.protocols import seeds_for
from baselines.sources import ROOT,upstream_imports,native_modules
from baselines.run_dirlab import make_jobs,summarize
from baselines.case_runner import main as case_main
from baselines.tests.test_adapters import config
from medical.dirlab_evaluation import score_landmarks


class SourceAlignmentTests(unittest.TestCase):
    def test_nir_coordinates_match_native_inference_helper(self):
        with upstream_imports('nir'):
            native_modules('nir')  # Imports only; no construction/training.
            utility=importlib.import_module('util')
            shape=(7,9,11)
            points=np.linspace(-.5,.5,27,dtype=np.float32).reshape(1,3,3,3)
            volume=np.arange(np.prod(shape),dtype=np.float32).reshape(shape)
            native=utility.warp_3d_volume(volume,points,shape,order=1)
            voxels=nir.native_output_voxels(points,shape,shape)
            ours=map_coordinates(volume,voxels.transpose(3,0,1,2),order=1,mode='constant',cval=0)
            np.testing.assert_allclose(ours,native,atol=1e-5)
            wrong=(points+1)*(np.asarray(shape)-1)/2
            self.assertGreater(np.max(np.abs(voxels-wrong)),.1)

    def test_ccidir_second_order_matches_native_math_both_directions(self):
        class Quadratic(torch.nn.Module):
            def __init__(self,a,b):
                super().__init__();self.a=a;self.b=b
            def forward(self,x):
                return self.a*x.square()+self.b*x+.01
        forward,reverse=Quadratic(.05,.02),Quadratic(-.03,.04)
        reg=SimpleNamespace(network=forward,network_rev=reverse,fit=Mock())
        shape=(17,19,23)
        fixed_points=np.tile([8.2,9.3,10.4],(300,1))
        moving_points=fixed_points+np.array([.2,.1,-.3])
        with tempfile.TemporaryDirectory() as directory, upstream_imports('ccidir'):
            models,general=native_modules('ccidir')
            case={}
            for key,phase,points in [('fixed','T00',fixed_points),('moving','T50',moving_points)]:
                np.savetxt(Path(directory)/f'{phase}_landmarks_voxel.csv',points,delimiter=',',header='x,y,z',comments='')
                case[key]={'data':np.zeros(shape,dtype=np.float32),'path':str(Path(directory)/f'{phase}.nii.gz'),'affine':np.eye(4)}
            zeros=torch.zeros
            def cpu_zeros(*args,**kwargs):
                if str(kwargs.get('device','')).startswith('cuda'): kwargs['device']='cpu'
                return zeros(*args,**kwargs)
            with patch.object(torch.Tensor,'cuda',lambda t:t),patch.object(torch,'zeros',cpu_zeros):
                coordinates=general.landmarks_to_coordmarks(fixed_points[:,::-1]+1,shape[::-1]).requires_grad_(True)
                a,_,offset=general.compute_coordinates_cycle(forward,reverse,coordinates,inf_order=2)
                native_prediction=general.coordmarks_to_landmarks(a+.5*offset,shape[::-1]).detach().numpy()[:,::-1]-1
                a1,_,offset1=general.compute_coordinates_cycle(forward,reverse,coordinates,inf_order=1)
                self.assertGreater(float((a+.5*offset-a1-.5*offset1).abs().max().detach()),1e-7)
                reverse_coordinates=general.landmarks_to_coordmarks(moving_points[:,::-1]+1,shape[::-1]).requires_grad_(True)
                r,_,ro=general.compute_coordinates_cycle(reverse,forward,reverse_coordinates,inf_order=2)
                native_reverse=general.coordmarks_to_landmarks(r+.5*ro,shape[::-1]).detach().numpy()[:,::-1]-1
                with patch.object(ccidir,'execution',return_value=nullcontext()),patch.object(ccidir,'native_modules',return_value=[SimpleNamespace(ImplicitRegistrator=Mock(return_value=reg)),general]),patch.object(ccidir,'load_mask',return_value=np.ones(shape,dtype=np.uint8)):
                    result=ccidir.run_case(case,config('ccidir'),'cpu')
            expected=score_landmarks(fixed_points,moving_points,native_prediction,np.eye(4))
            expected_reverse=score_landmarks(moving_points,fixed_points,native_reverse,np.eye(4))
            self.assertAlmostEqual(result['metrics']['continuous_tre_mm']['mean_mm'],expected['continuous_tre_mm']['mean_mm'])
            self.assertAlmostEqual(result['metrics']['reverse_T50_to_T00']['continuous_tre_mm']['mean_mm'],expected_reverse['continuous_tre_mm']['mean_mm'])
            self.assertEqual(result['metrics']['inference_order'],2)
            reg.fit.assert_called_once()  # Mock only.

    def test_seed_policy_and_unique_ccidir_paths(self):
        self.assertEqual(seeds_for('ours','dirlab'),[42])
        self.assertEqual(seeds_for('ccidir','dirlab'),list(range(50)))
        for dataset in ('abdomenctct','oasis'):
            self.assertEqual(seeds_for('ccidir',dataset),[42])
        with patch('baselines.run_dirlab.verify_source'):
            jobs=make_jobs(['ccidir'],[1],Path('/missing'),Path('/unused'),ROOT/'configs')
        self.assertEqual(len(jobs),50)
        self.assertEqual(len({j['output'] for j in jobs}),50)
        self.assertEqual({j['seed'] for j in jobs},set(range(50)))
        self.assertTrue(all(j['command'][2]=='baselines.case_runner' for j in jobs))

    def test_seed_override_reaches_saved_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config.json';path.write_text(json.dumps(config('ccidir')))
            with patch('baselines.case_runner.run_case',return_value={}) as run,patch('baselines.case_runner.importlib.import_module',return_value=SimpleNamespace(run_case=Mock())),redirect_stdout(io.StringIO()):
                case_main(['--method','ccidir','--config',str(path),'--moving','moving','--fixed','fixed','--output',directory,'--seed','17'])
            self.assertEqual(run.call_args.args[2]['seed'],17)
            self.assertEqual(json.loads(path.read_text())['seed'],42)

    def test_failed_fits_stay_in_robustness_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'success';output.mkdir()
            (output/'metrics.json').write_text(json.dumps({'idir_rounded_tre_mm':{'mean_mm':3},'reverse_T50_to_T00':{'idir_rounded_tre_mm':{'mean_mm':1}}}))
            rows=[{'method':'ccidir','case':1,'returncode':0,'output':str(output)},
                  {'method':'ccidir','case':1,'returncode':1,'output':str(Path(directory)/'failed')}]
            result=summarize(rows)['ccidir']
            self.assertEqual(result['intended_direction_results'],4)
            self.assertEqual(result['scored_direction_results'],2)
            self.assertEqual(result['unscored_direction_results'],2)
            self.assertEqual(result['measured_TRE_failures'],1)
            self.assertEqual(result['measured_failure_fraction'],.5)

if __name__=='__main__': unittest.main()
