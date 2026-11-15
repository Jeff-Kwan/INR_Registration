"""Synthetic Learn2Reg metric/adapter tests; optimizers and recovery are mocked."""
import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch,Mock
from types import SimpleNamespace
import unittest
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import map_coordinates
from medical import abdomen_evaluation as evaluation,abdomen_method
from medical.abdomen_experiments import main,make_jobs,METHODS
from medical.configuration import load_config
from medical.registration.geometry import identity_grid
from medical.case_runner import run_case
from baselines import convexadam
from baselines.tests.test_adapters import config
from contextlib import nullcontext

class TestAbdomen(unittest.TestCase):
    dataset="abdomenctct"
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        (self.root/'imagesTr').mkdir();(self.root/'labelsTr').mkdir()
        self.shape=(9,8,7)
        identity=np.stack(np.meshgrid(*(np.arange(n) for n in self.shape),indexing='ij'),-1).astype(np.float32)
        self.identity=identity
        self.mapping=identity+np.array([1,0,0],dtype=np.float32)
        self.label_count=13 if self.dataset=='abdomenctct' else 35
        moving=((identity[...,0]+2*identity[...,1]+3*identity[...,2])%self.label_count+1).astype(np.uint8)
        fixed=map_coordinates(moving,self.mapping.transpose(3,0,1,2),order=0)
        self.case={}
        pair=evaluation.PROTOCOLS[self.dataset]['eval_pairs'][0]
        for key,label in [('fixed',fixed),('moving',moving)]:
            name=Path(pair[key]).name
            data=np.arange(np.prod(self.shape),dtype=np.float32).reshape(self.shape)
            affine=np.diag([2.,2.,2.,1.])
            nib.save(nib.Nifti1Image(data,affine),self.root/'imagesTr'/name)
            nib.save(nib.Nifti1Image(label,affine),self.root/'labelsTr'/name)
            self.case[key]={'path':str(self.root/'imagesTr'/name),'data':data,'affine':affine}

    def test_official_direction_and_metrics(self):
        scores=evaluation.evaluate(self.case,self.mapping,protocol=self.dataset)
        self.assertAlmostEqual(scores['DSC']['mean'],1)
        self.assertEqual(scores['HD95']['mean'],0)
        self.assertAlmostEqual(scores['LogJacDetStd']['mean'],0)
        self.assertEqual(scores['folding_fraction'],0)
        self.assertLess(evaluation.evaluate(self.case,self.identity,protocol=self.dataset)['DSC']['mean'],1)
        json.dumps(scores,allow_nan=False)

    def test_callable_chunks_and_shifted_sdlogj(self):
        mapping=self.identity.copy();mapping[...,0]+=.03*mapping[...,0]**2
        scores=evaluation.evaluate(self.case,mapping,protocol=self.dataset)
        native=evaluation.official_functions()
        jac=native.jacobian_determinant((mapping-self.identity)[None].transpose(0,4,1,2,3))
        self.assertAlmostEqual(scores['LogJacDetStd']['mean'],np.log((jac+3).clip(1e-9,1e9)).std())
        from_callable=evaluation.evaluate(self.case,lambda points: points+np.array([1,0,0]),protocol=self.dataset)
        self.assertEqual(from_callable['DSC']['mean'],1)

    def test_parity_with_original_official_evaluator(self):
        # Execute the pinned evaluator function with its displacement loader mocked.
        # Only temporary labels/config/score JSON are written; no map files exist.
        import ast,os
        from datetime import datetime
        from collections import OrderedDict
        source=ast.parse((evaluation.PROTOCOL_DIR/'l2r_evaluation.py').read_text())
        node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_L2R')
        metrics=evaluation.official_functions()
        mapping=self.identity.copy();mapping[...,0]+=.03*mapping[...,0]**2
        namespace={'json':json,'np':evaluation.NumpyCompat(),'nib':nib,'map_coordinates':map_coordinates,
                   'os':SimpleNamespace(path=SimpleNamespace(join=os.path.join,isdir=os.path.isdir,isfile=lambda path:True)),
                   'datetime':datetime,'OrderedDict':OrderedDict,'file_exists':lambda *args:True,
                   'load_disp':Mock(return_value=mapping-self.identity),**vars(metrics)}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'official_evaluator','exec'),namespace)
        protocol=json.loads(json.dumps(evaluation.PROTOCOLS[self.dataset]))
        protocol['eval_pairs']=protocol['eval_pairs'][:1];protocol['expected_shape']=[*self.shape,3]
        config_path=self.root/'config.json';config_path.write_text(json.dumps(protocol))
        output=self.root/'official_scores.json'
        with redirect_stdout(io.StringIO()):
            namespace['evaluate_L2R']('no_map_files',str(self.root),str(output),str(config_path))
        original=next(iter(json.loads(output.read_text())['cases'].values()))
        ours=evaluation.evaluate(self.case,mapping,protocol=self.dataset)
        for name in ('DSC','HD95','LogJacDetStd','num_foldings'):
            np.testing.assert_allclose(ours[name]['mean'],original[name]['mean'],atol=1e-12)
            if name in ('DSC','HD95'):
                np.testing.assert_allclose(ours[name]['detailed'],original[name]['detailed'],atol=1e-12)

    def test_missing_label_is_explicit_null(self):
        path=evaluation.label_path(self.case['fixed']);image=nib.load(path)
        labels=np.asarray(image.dataobj).copy();labels[labels==self.label_count]=0
        nib.save(nib.Nifti1Image(labels,image.affine),path)
        scores=evaluation.evaluate(self.case,self.mapping,protocol=self.dataset)
        self.assertIsNone(scores['DSC']['detailed'][self.label_count-1])
        self.assertIsNone(scores['HD95']['detailed'][self.label_count-1])
        json.dumps(scores,allow_nan=False)

    def test_empty_prediction_retains_official_infinite_hd95(self):
        import surface_distance
        truth=np.zeros((5,5,5),dtype=bool);truth[2,2,2]=True
        distances=surface_distance.compute_surface_distances(truth,np.zeros_like(truth),np.ones(3))
        distance=surface_distance.compute_robust_hausdorff(distances,95.)
        self.assertTrue(np.isinf(distance))
        self.assertIsNone(evaluation.json_metric(distance))

    def test_ours_runner_without_training_or_label_inputs(self):
        pair=torch.nn.Linear(1,1);pair.model_config={'width':1}
        # Torch map is ZYX with XYZ vector components.
        mapping=torch.from_numpy(self.mapping.transpose(2,1,0,3).copy())
        recovered=SimpleNamespace(inverse=mapping,diagnostics=[])
        with (patch.object(abdomen_method,'fit_pair',return_value=pair) as fit,
              patch.object(abdomen_method,'recover',return_value=recovered),
              patch.object(abdomen_method,'endpoint_reconstruction',return_value={})):
            scores=run_case(self.case['moving']['path'],self.case['fixed']['path'],load_config(),self.root/'output',__import__('functools').partial(abdomen_method.run_case,protocol=self.dataset),'cpu')
        self.assertEqual(scores['DSC']['mean'],1)
        self.assertNotIn('moving_mask',fit.call_args.kwargs)
        self.assertEqual(set(p.name for p in (self.root/'output').iterdir()),{'checkpoint.pt','metrics.json','run.json'})

    def test_convexadam_uses_official_evaluation(self):
        cfg=config('convexadam');cfg['evaluation']=self.dataset
        native=SimpleNamespace(convex_adam_pt=Mock(return_value=self.mapping-self.identity))
        with patch.object(convexadam,'upstream_imports',return_value=nullcontext()),patch.object(convexadam,'execution',return_value=nullcontext(torch.device('cpu'))),patch.object(convexadam,'native_modules',return_value=[native]):
            result=convexadam.run_case(self.case,cfg,'cpu')
        self.assertEqual(result['metrics']['DSC']['mean'],1)
        self.assertFalse(result['checkpoint']['parameter_checkpoint'])

    def test_fireants_and_nir_abdomen_routes(self):
        from baselines.tests.test_adapters import AdapterTests
        from baselines import fireants,nir
        for name,module in [('fireants',fireants),('nir',nir)]:
            with self.subTest(method=name):
                harness=AdapterTests()
                harness.setUp()
                try:
                    harness.shape=self.shape;harness.case=self.case;harness.points=np.zeros((300,3));harness.shift=np.array([1.,0.,0.])
                    cfg=config(name);cfg['evaluation']=self.dataset
                    if name=='fireants':
                        reg=SimpleNamespace(optimize=Mock(),get_warped_coordinates=Mock(return_value=harness.grid()))
                        image=SimpleNamespace(Image=SimpleNamespace(load_file=Mock()),BatchedImages=lambda images:images)
                        native=[image,SimpleNamespace(SyNRegistration=Mock(return_value=reg))]
                        harness.enterContext(patch.dict('sys.modules',{'fireants.interpolator':SimpleNamespace(fireants_interpolator=SimpleNamespace(use_ffo=False))}))
                    else:
                        native=harness.nir_native()
                        harness.enterContext(patch.object(torch.Tensor,'cuda',lambda value:value))
                    harness.contexts(module,native)
                    scores=module.run_case(self.case,cfg,'cpu')['metrics']
                    self.assertEqual(scores['DSC']['mean'],1)
                finally:
                    harness.doCleanups()

    def test_ccidir_dense_consensus_abdomen_route(self):
        from baselines import ccidir
        cfg=config('ccidir');cfg['evaluation']=self.dataset
        net=torch.nn.Linear(3,3)
        reg=SimpleNamespace(network=net,network_rev=torch.nn.Linear(3,3),fit=Mock())
        def consensus(net,reverse,coords,inf_order):
            shift=torch.tensor([0.,0.,4./(self.shape[0]-1)],dtype=coords.dtype)
            return coords,coords,shift.expand_as(coords)
        native=[SimpleNamespace(ImplicitRegistrator=Mock(return_value=reg)),SimpleNamespace(compute_coordinates_cycle_autoretry=Mock(side_effect=consensus))]
        with patch.object(ccidir,'upstream_imports',return_value=nullcontext()),patch.object(ccidir,'execution',return_value=nullcontext()),patch.object(ccidir,'native_modules',return_value=native),patch.object(torch.Tensor,'cuda',lambda value:value):
            result=ccidir.run_case(self.case,cfg,'cpu')
        self.assertEqual(result['metrics']['DSC']['mean'],1)
        kwargs=native[0].ImplicitRegistrator.call_args.kwargs
        self.assertTrue(np.all(kwargs['mask']==1))
        self.assertTrue(np.all(kwargs['mask_2']==1))

    def test_plan_has_exact_directed_pairs_and_no_execution(self):
        jobs=make_jobs(list(METHODS),list(range(1,len(evaluation.PROTOCOLS[self.dataset]['eval_pairs'])+1)),self.root,self.root/'output',self.dataset)
        self.assertEqual(len(jobs),len(evaluation.PROTOCOLS[self.dataset]['eval_pairs'])*5)
        self.assertEqual(len({j['output'] for j in jobs}),len(jobs))
        with patch('subprocess.run') as run,redirect_stdout(io.StringIO()) as stream:
            main(['--data',str(self.root),'--output',str(self.root/'planned')],dataset=self.dataset)
        run.assert_not_called();self.assertFalse((self.root/'planned').exists())
        self.assertFalse(json.loads(stream.getvalue())['execution'])

if __name__=='__main__': unittest.main()
