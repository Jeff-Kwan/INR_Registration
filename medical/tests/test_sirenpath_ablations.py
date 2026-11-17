"""CPU-only analytical and mocked execution checks; no training or real-data recovery."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from medical.configuration import load_config, validate_config
from medical.models.siren import PairedSiren, coordinates_from_indices
from medical.registration.objective import TrainingBatch, endpoint_loss, parameter_agreement
from medical.registration.recovery import recover, derivative_grid, RecoveryResult
from medical.registration.geometry import identity_grid
from medical.registration.intensity_crossfade import IntensityCrossfade
from medical.sirenpath_reconstruction import endpoint_scores, patch_ncc
from medical.sirenpath_ablations import settings_for, make_plan, aggregate, write_report, main, require_idle_gpus, ROOT


class TestSirenPathAblations(unittest.TestCase):
    def test_exact_factorial_and_canonical_unchanged(self):
        for run, endpoint, transport, registration in ((1,'mse',False,False),(2,'mse',True,False),
                (3,'mse',False,True),(4,'mse',True,True),(5,'ncc',True,True)):
            c=settings_for(run);o=c['objective']
            self.assertEqual(o['endpoint'],endpoint)
            self.assertEqual(o['weight']>0,transport)
            self.assertEqual(o['registration']['weight']>0,registration)
            self.assertEqual(c['training']['steps'],5000)
            self.assertEqual(o['ncc']['moment_weight'],.005 if endpoint=='ncc' else 0.)
            self.assertEqual(o['parameter_l2'],.01)
            self.assertEqual(c['training']['registration_weight_schedule']['start_weight'],.03 if registration else 0.)
            if endpoint=='mse':
                with self.assertRaises(ValueError):validate_config(c)
        self.assertEqual(settings_for(5),load_config())

    def test_mse_and_live_parameter_gradients(self):
        torch.manual_seed(42)
        pair=PairedSiren(width=4,layers=1).double()
        with torch.no_grad():pair.target.net[-1].bias.add_(.1)
        x=torch.randn(17,3,dtype=torch.float64)
        batch=TrainingBatch(x,torch.rand(17,dtype=torch.float64),torch.rand(17,dtype=torch.float64))
        terms=endpoint_loss(pair,batch,settings_for(1)['objective'])
        e=pair.prepare(temporal=False)
        expected=((pair.source(x).flatten()-batch.source).square().mean()
                  +(pair.target(x).flatten()-batch.target).square().mean()
                  +.01*parameter_agreement(e.source,e.target))
        torch.testing.assert_close(terms['total'],expected)
        a=torch.autograd.grad(terms['total'],tuple(pair.parameters()),retain_graph=True)
        b=torch.autograd.grad(expected,tuple(pair.parameters()))
        for actual,truth in zip(a,b):torch.testing.assert_close(actual,truth)

    def test_disabled_transport_skips_derivatives_and_decoder(self):
        from medical.registration.captured_objective import joint_loss
        pair=PairedSiren(width=4,layers=1).double()
        e=pair.prepare(temporal=False);x=torch.randn(13,3,dtype=torch.float64)
        y=torch.rand(13,dtype=torch.float64)
        with patch('medical.registration.captured_objective._parameter_direction',side_effect=AssertionError('disabled')):
            actual,finite=joint_loss(e.source,e.target,e.omegas,x,x,y,y,None,None,None,None,None,
                None,None,None,None,None,None,None,None,0.,0.,dict(epsilon=1e-8,moment_weight=0.),
                endpoint_mode='mse',transport_enabled=False,registration_enabled=False)
        expected=endpoint_loss(pair,TrainingBatch(x,y,y),dict(endpoint='mse',ncc=dict(moment_weight=0.)))
        torch.testing.assert_close(actual['total'],expected['total']);self.assertTrue(finite)

    def test_loss_switches_preserve_enabled_loss_gradients(self):
        from medical.registration.captured_objective import joint_loss
        from medical.registration.symmetric_registration import SymmetricPairRegistration
        torch.manual_seed(42)
        image=torch.rand(11,13,15,dtype=torch.float64)
        pair=PairedSiren(width=4,layers=2,omega=10.).double()
        with torch.no_grad():
            for parameter in pair.target.parameters():parameter.add_(.001*torch.randn_like(parameter))
        branch=SymmetricPairRegistration(image,image.flip(0),torch.eye(4),grid_size=2,particles=31,
            pairs_per_cell=2,crop_fraction=.75,random_timesteps=False,compile_derivatives=False,
            compile_velocity=False,gaussian_kernel='three_tap',training_integrator='rk2')
        x=coordinates_from_indices(torch.arange(17),image.shape).double()
        batch=TrainingBatch(x,image.flatten()[:17],branch.fixed.flatten()[:17])
        positions=branch.sample_positions()
        reference=branch.loss(pair,positions=positions)
        e=pair.prepare(temporal=False)
        for run in (2,3,4):
            o=settings_for(run)['objective']
            weight,reg=o['weight'],o['registration']['weight']
            expected=(endpoint_loss(pair,batch,o)['total']+weight*reference['grid_pdlk']
                      +reg*reference['registration'])
            actual,finite=joint_loss(e.source,e.target,e.omegas,x,x,batch.source,batch.target,
                branch.coordinates,branch.spatial_scale,positions,branch.moving,branch.fixed,
                branch.origin,branch.domain_extent,branch.extent,branch.sampled_times,tuple(branch.sigma),
                branch.grid_size,branch.c,branch.eta,weight,reg,dict(epsilon=1e-8,moment_weight=0.),
                parameter_l2=.01,endpoint_mode='mse',transport_enabled=weight>0,registration_enabled=reg>0)
            self.assertTrue(finite)
            torch.testing.assert_close(actual['total'],expected,rtol=1e-8,atol=1e-10)
            a=torch.autograd.grad(actual['total'],tuple(pair.parameters()),retain_graph=True)
            b=torch.autograd.grad(expected,tuple(pair.parameters()),retain_graph=True)
            for actual,truth in zip(a,b):torch.testing.assert_close(actual,truth,rtol=1e-7,atol=1e-9)

    def test_raw_finite_difference_axis_and_coarse_scaling(self):
        z,y,x=torch.meshgrid(torch.arange(7),torch.arange(8),torch.arange(9),indexing='ij')
        a=.03*x.double()+.04*y.double()+.05*z.double();b=2*a+.1
        path=IntensityCrossfade(a,b)
        for shape in (a.shape,(4,5,6)):
            g,q=derivative_grid(path,shape,.25,chunk=11,compile_derivatives=False)
            ratio=torch.tensor([(n-1)/(m-1) for n,m in zip(reversed(a.shape),reversed(shape))])
            expected=torch.tensor([.03,.04,.05]).double()*1.25*ratio
            torch.testing.assert_close(g,expected.expand_as(g),rtol=2e-6,atol=1e-8)
            self.assertTrue(torch.isfinite(q).all())
        # At an exact native voxel the temporal derivative is fixed-minus-moving.
        coords=coordinates_from_indices(torch.arange(a.numel()),a.shape).double()
        scale=torch.tensor([2/(n-1) for n in reversed(a.shape)]).double()
        _,q=path.prepare().at(.9).derivatives(coords,scale)
        torch.testing.assert_close(q,(b-a).flatten(),atol=1e-7,rtol=1e-6)

    def test_raw_shared_recovery_translation_and_cleanup(self):
        z,y,x=torch.meshgrid(torch.arange(7),torch.arange(8),torch.arange(9),indexing='ij')
        a=.03125*(x+y+z)
        path=IntensityCrossfade(a,a+.015625)
        settings=load_config()['recovery']
        with tempfile.TemporaryDirectory() as tmp:
            result=recover(path,a.shape,settings,tmp,chunk=31,compile_derivatives=False,compile_solver=False,max_axis=256)
            self.assertEqual(list(Path(tmp).iterdir()),[])
        gradient=torch.tensor([.03125,.03125,.03125])
        v=-.015625*gradient/(gradient.square().sum()+settings['ridge'])
        grid=identity_grid(a.shape,'cpu')
        torch.testing.assert_close(result.forward,grid+v,rtol=2e-5,atol=2e-5)
        torch.testing.assert_close(result.inverse,grid-v,rtol=2e-5,atol=2e-5)
        with tempfile.TemporaryDirectory() as tmp:
            with patch('medical.registration.recovery.derivative_grid',side_effect=RuntimeError('injected')):
                with self.assertRaisesRegex(RuntimeError,'injected'):
                    recover(path,a.shape,settings,tmp,compile_solver=False)
            self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_patch_ncc_and_raw_endpoint_json(self):
        torch.manual_seed(42);a=torch.rand(11,12,13)
        self.assertAlmostEqual(patch_ncc(2*a+.3,a)['mean'],1.,places=6)
        self.assertAlmostEqual(patch_ncc(-a,a)['mean'],-1.,places=6)
        self.assertEqual(patch_ncc(torch.zeros_like(a),a)['mean'],0.)
        self.assertIsNone(patch_ncc(a,torch.zeros_like(a))['mean'])
        scores=endpoint_scores(None,(a,a.flip(0)),raw=True)
        self.assertTrue(scores['PSNR']['positive_infinity'])
        self.assertIsNone(scores['PSNR']['mean'])
        self.assertAlmostEqual(scores['PatchNCC']['mean'],1.,places=7)
        json.dumps(scores,allow_nan=False)

    def test_raw_adapter_cannot_fit_or_serialize_images(self):
        from medical.raw_crossfade_method import run_case
        a=np.arange(11**3,dtype=np.float32).reshape(11,11,11)
        case={k:dict(data=a.copy(),affine=np.eye(4),path=f'/fake/{k}.nii') for k in ('moving','fixed')}
        grid=identity_grid(a.shape,'cpu')
        scores=dict(DSC=dict(mean=1.),HD95=dict(mean=0.),LogJacDetStd=dict(mean=0.),folding_fraction=0.)
        with (patch('medical.sirenpath_ablation_method.validate_case'),
             patch('medical.sirenpath_ablation_method.fit_pair',side_effect=AssertionError('No fitting')),
             patch('medical.sirenpath_ablation_method.recover',return_value=RecoveryResult(grid,grid)), \
             patch('medical.sirenpath_ablation_method.evaluate',return_value=scores)):
            result=run_case(case,load_config(),'cpu')
        self.assertEqual(result['checkpoint']['kind'],'provenance_only')
        self.assertEqual(result['metrics']['timings']['training_seconds'],0.)
        self.assertEqual(result['metrics']['NPJ']['mean'],0.)
        self.assertEqual(result['metrics']['SDlogJ']['mean'],0.)
        json.dumps(result['checkpoint'],allow_nan=False)
        json.dumps(result['metrics'],allow_nan=False)

    def test_real_plan_is_side_effect_free_and_quick4_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'absent'
            plan=make_plan(ROOT/'Data/OASIS',output,list(range(1,8)),list(range(1,20)),[0])
            self.assertEqual(plan['total_jobs'],133);self.assertEqual(plan['training_jobs'],95)
            self.assertFalse(output.exists())
            self.assertFalse(any(j['missing_inputs'] for j in plan['jobs']))
            self.assertTrue(all(j['training_steps']==0 for j in plan['jobs'] if j['run']==6))
            with patch('medical.sirenpath_ablations.GPUWorker',side_effect=AssertionError('No workers')),redirect_stdout(io.StringIO()):
                subset=main(['--output',str(output),'--subset','quick4'])
            self.assertEqual(subset['pair_indices'],[1,4,8,9]);self.assertFalse(output.exists())
            with self.assertRaises(ValueError):make_plan(ROOT/'Data/OASIS',ROOT/'Data/OASIS/output',[1],[1],[0])

    def test_reconstruction_control_reuses_frozen_model_without_rescaling(self):
        from medical.sirenpath_ablation_method import run_case, load_canonical_checkpoint
        from medical.sirenpath_reconstruction import reconstruct_endpoint
        c=load_config();c['model'].update(width=4,layers=1)
        torch.manual_seed(42);model=PairedSiren(3,**c['model'])
        # Preserve fitted affine intensity differences, including values outside [0,1].
        with torch.no_grad():
            model.source.net[-1].bias.fill_(-.4)
            model.target.net[-1].bias.fill_(1.4)
        a=np.arange(11**3,dtype=np.float32).reshape(11,11,11)
        case={k:dict(data=a.copy(),affine=np.eye(4),path=f'/fake/{k}.nii') for k in ('moving','fixed')}
        arrays=[torch.from_numpy((a/a.max()).transpose(2,1,0).copy()) for _ in range(2)]
        model.eval().requires_grad_(False)
        expected=endpoint_scores(model,arrays)
        reconstructions=tuple(reconstruct_endpoint(m,image) for m,image in zip((model.source,model.target),arrays))
        self.assertLess(float(reconstructions[0].min()),0.)
        self.assertGreater(float(reconstructions[1].max()),1.)
        grid=identity_grid(a.shape,'cpu')
        scores=dict(DSC=dict(mean=1.),HD95=dict(mean=0.),LogJacDetStd=dict(mean=0.),folding_fraction=0.)
        def recovery(path,*args,**kwargs):
            torch.testing.assert_close(path.temporal[...,0],reconstructions[1]-reconstructions[0])
            return RecoveryResult(grid,grid)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);checkpoint=root/'checkpoint.pt'
            evidence=dict(moving=case['moving']['path'],fixed=case['fixed']['path'],shape=list(a.shape),affine=np.eye(4).tolist(),config=c)
            (root/'run.json').write_text(json.dumps(evidence))
            (root/'metrics.json').write_text(json.dumps(dict(timings=dict(training_seconds=17.))))
            torch.save(dict(model=model.model_config,config=c,state=model.state_dict(),training=dict(spatial_units='voxel')),checkpoint)
            frozen,reference=load_canonical_checkpoint(checkpoint,case,c,'cpu')
            self.assertFalse(any(p.requires_grad for p in frozen.parameters()))
            with patch('medical.sirenpath_ablation_method.validate_case'), \
                 patch('medical.sirenpath_ablation_method.fit_pair',side_effect=AssertionError('No retraining')), \
                 patch('medical.sirenpath_ablation_method.recover',side_effect=recovery), \
                 patch('medical.sirenpath_ablation_method.evaluate',return_value=scores):
                result=run_case(case,c,'cpu',reference_checkpoint=checkpoint)
            self.assertEqual(result['metrics']['PSNR'],expected['PSNR'])
            self.assertEqual(result['metrics']['PatchNCC'],expected['PatchNCC'])
            self.assertEqual(result['metrics']['timings']['training_seconds'],0.)
            self.assertEqual(result['metrics']['reused_training_seconds'],17.)
            self.assertEqual(result['checkpoint']['kind'],'provenance_only')
            self.assertEqual(result['checkpoint']['control'],'reconstructed_crossfade')
            json.dumps(result['checkpoint'],allow_nan=False)
            mismatch=dict(case);mismatch['moving']=dict(case['moving'],path='/different.nii')
            with self.assertRaisesRegex(ValueError,'different image pair'):
                load_canonical_checkpoint(checkpoint,mismatch,c,'cpu')
            altered=json.loads(json.dumps(c));altered['training']['lr_final_factor']=.25
            with self.assertRaisesRegex(ValueError,'identical canonical'):
                load_canonical_checkpoint(checkpoint,case,altered,'cpu')

    def test_reconstruction_control_waits_for_canonical_and_never_refits_failures(self):
        calls=[]
        class FakeWorker:
            def __init__(self,gpu):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def run(self,command,log):
                output=Path(command[command.index('--output')+1]);output.mkdir()
                run=int(output.parent.name[:2]);index=int(output.name[-3:]);calls.append((run,index))
                if run==5 and index==4:return dict(returncode=1)
                scores={k:dict(mean=.5) for k in ('PSNR','PatchNCC','DSC','HD95','NPJ','SDlogJ')}
                (output/'metrics.json').write_text(json.dumps(scores))
                (output/'checkpoint.pt').write_text('synthetic dependency marker')
                (output/'run.json').write_text('{}')
                return dict(returncode=0)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'study'
            with patch('medical.sirenpath_ablations.GPUWorker',FakeWorker), \
                 patch('medical.sirenpath_ablations.require_idle_gpus'),redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(['--output',str(output),'--runs','7','5','--pairs','1','4','--gpus','0','--run'])
            self.assertEqual(calls,[(5,1),(5,4),(7,1)])
            summary=json.loads((output/'summary.json').read_text())
            failed=[r for r in summary['jobs'] if r['run']==7 and r['pair_index']==4][0]
            self.assertIn('dependency',failed['error'])
            plan=json.loads((output/'plan.json').read_text())
            cfg=json.loads((output/'configs/07_reconstructed_crossfade.json').read_text())
            self.assertEqual(cfg['canonical_checkpoint_root'],str(output/'05_sirenpath'))
            self.assertEqual(plan['training_jobs'],2)
        with self.assertRaisesRegex(ValueError,'requires run 5'):
            make_plan(ROOT/'Data/OASIS',Path('/tmp/standalone-crossfade'),[7],[1],[0])

    def test_busy_gpu_blocks_before_any_worker(self):
        from types import SimpleNamespace
        with patch('medical.sirenpath_ablations.subprocess.run',return_value=SimpleNamespace(stdout='123\n')):
            with self.assertRaisesRegex(ValueError,'active compute jobs'):require_idle_gpus([0])

    def test_mocked_launcher_completes_and_preserves_failures(self):
        class FakeWorker:
            def __init__(self,gpu):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def run(self,command,log):
                output=Path(command[command.index('--output')+1]);output.mkdir()
                config=Path(command[command.index('--config')+1]);self_config=json.loads(config.read_text())
                assert self_config['training']['steps']==5000
                Path(log).write_text('synthetic dispatcher test')
                if output.name=='pair004':return dict(returncode=1)
                scores={k:dict(mean=.5) for k in ('PSNR','PatchNCC','DSC','HD95','NPJ','SDlogJ')}
                (output/'metrics.json').write_text(json.dumps(scores))
                return dict(returncode=0)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'run'
            with patch('medical.sirenpath_ablations.GPUWorker',FakeWorker), \
                 patch('medical.sirenpath_ablations.require_idle_gpus'),redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(['--output',str(output),'--runs','1','6','--pairs','1','4','--gpus','0','--run'])
            self.assertEqual(error.exception.code,1)
            summary=json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['finished'],4)
            self.assertEqual(summary['runs']['1']['successful'],1)
            self.assertEqual(summary['runs']['6']['failed'],1)
            self.assertTrue((output/'ABLATIONS_REPORT.md').is_file())
            self.assertEqual(len(list((output/'configs').glob('*.json'))),2)

    def test_aggregate_counts_failures_and_infinite_psnr(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'case';p.mkdir()
            scores={k:dict(mean=.5) for k in ('PSNR','PatchNCC','DSC','HD95','NPJ','SDlogJ')}
            scores['PSNR']=dict(mean=None,positive_infinity=True);scores['HD95']['mean']=None
            (p/'metrics.json').write_text(json.dumps(scores))
            records=[dict(run=6,output=str(p),returncode=0,seconds=12.),dict(run=6,output=str(p),returncode=1,seconds=3.)]
            summary=aggregate(records,[dict(run=6),dict(run=6)])
            self.assertEqual(summary['6']['successful'],1);self.assertEqual(summary['6']['failed'],1)
            self.assertEqual(summary['6']['metrics']['PSNR']['infinite_pairs'],1)
            self.assertEqual(summary['6']['metrics']['HD95']['scored_pairs'],0)
            self.assertEqual(summary['6']['metrics']['Runtime']['mean'],12.)
            write_report(Path(tmp),dict(runs=summary));self.assertIn('∞',(Path(tmp)/'ABLATIONS_REPORT.md').read_text())
            json.dumps(summary,allow_nan=False)
