"""Synthetic physical geometry checks: no optimizer or saved-model recovery."""
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from medical.working_grid import (geometry,canonical_geometry,resample,prepare,predictor,
                                 original_reconstruction,restore_checkpoint,validate_metadata)
from medical.registration.geometry import identity_grid
from medical.configuration import load_config,validate_config
from medical.models.siren import PairedSiren,endpoint_reconstruction


def case(data,affine):
    return {key:{'data':data.copy(),'affine':affine.copy(),'path':key} for key in ('moving','fixed')}


class TestWorkingGrid(unittest.TestCase):
    def test_geometry_frozen_reference_axes_and_rounding(self):
        a=np.eye(4);a[:3,:3]=[[0,-2,0],[.5,0,0],[0,0,-1.5]];a[:3,3]=[7,-3,12]
        ref=geometry([9,5,5],a);w=canonical_geometry(ref)
        self.assertEqual(w['shape'],[5,9,7])
        b=np.asarray(w['affine'])
        np.testing.assert_allclose(b[:3,:3],[[0,-1,0],[1,0,0],[0,0,-1]])
        np.testing.assert_allclose(a[:3,:3]@np.array([4,2,2])+a[:3,3],b[:3,:3]@np.array([2,4,3])+b[:3,3])
        snap=np.eye(4);snap[0,0]=1+1e-8
        self.assertEqual(canonical_geometry(geometry([5,5,5],snap))['shape'],[5,5,5])
        bad=a.copy();bad[0,0]=.1
        with self.assertRaises(ValueError):geometry([5]*3,bad)

    def test_resampling_fields_masks_antialias_and_bypass(self):
        xyz=np.indices((5,7,9),dtype=np.float32);data=xyz[0]+2*xyz[1]+3*xyz[2]
        ref=geometry(list(data.shape),np.eye(4))
        out,details=resample(data,np.eye(4),ref)
        self.assertIs(out,data);self.assertEqual(details['interpolation'],'bypass')
        a=np.diag([2.,2.,2.,1.]);coarse=data[::2,::2,::2]
        out,_=resample(coarse,a,ref)
        np.testing.assert_allclose(out,data,atol=1e-6)
        constant=np.ones((9,13,17),np.float32);a=np.diag([.5,.5,.5,1.])
        out,d=resample(constant,a,ref);np.testing.assert_allclose(out,1)
        np.testing.assert_allclose(d['antialias_sigma_source_voxels'],[.5*np.sqrt(3)]*3)
        mask=np.ones((5,7,9),np.uint8);shift=np.eye(4);shift[0,3]=2
        out,_=resample(mask,shift,ref,mask=True)
        self.assertFalse(out[:2].any());self.assertTrue(out[2:].all())

    def test_smooth_phantom_multiple_sampling_rates(self):
        ref=geometry([17]*3,np.eye(4));errors=[]
        for spacing in (.5,1.,2.):
            shape=[int(16/spacing)+1]*3
            x=np.indices(shape,dtype=np.float32)*spacing
            values=np.cos(x[0]/10)+np.sin(x[1]/12)+.02*x[2]
            out,_=resample(values,np.diag([spacing]*3+[1]),ref)
            y=np.indices((17,)*3,dtype=np.float32)
            truth=np.cos(y[0]/10)+np.sin(y[1]/12)+.02*y[2]
            errors.append(float(np.sqrt(np.mean((out[2:-2,2:-2,2:-2]-truth[2:-2,2:-2,2:-2])**2))))
        self.assertLess(max(errors),.004)

    def test_prepare_frozen_geometry_normalization_and_axes(self):
        xyz=np.indices((5,7,9),dtype=np.float32);data=xyz[0]+xyz[1]+xyz[2]
        ref=geometry(list(data.shape),np.eye(4));options={'spacing_mm':1.,'reference_geometry':ref}
        a,affine,_,meta=prepare(case(data,np.eye(4)),options,'cpu')
        b,_,_,other=prepare(case(data[::2,::2,::2],np.diag([2.,2.,2.,1.])),options,'cpu')
        self.assertEqual(meta['working_geometry'],other['working_geometry'])
        torch.testing.assert_close(a[0],b[0]);self.assertEqual(tuple(a[0].shape),(9,7,5))
        np.testing.assert_allclose(affine[:3,:3],np.eye(3)[:,::-1])
        self.assertEqual(meta['endpoints']['moving']['normalization'],{'min':0.,'max':18.})
        with self.assertRaises(ValueError):prepare(case(data,np.eye(4)),options,'cpu',[np.zeros_like(data)]*2)

    def test_displacement_identity_translation_and_roundtrip(self):
        a=np.eye(4);a[:3,:3]=[[0,-2,0],[1.5,0,0],[0,0,-.5]];a[:3,3]=[7,9,-4]
        ref=geometry([5,7,9],a);w=canonical_geometry(ref);meta={'working_geometry':w}
        grid=identity_grid(tuple(reversed(w['shape'])),'cpu')
        points=np.array([[-.1,0,0],[0,0,0],[2,3,4],[4,6,8],[4.1,6,8]])
        for sign in (1,-1):
            delta=np.array([.2,-.3,.1],np.float32)*sign
            p=predictor(grid+torch.tensor(delta),meta,a,a)
            expected=points+delta@np.linalg.solve(a[:3,:3],np.asarray(w['affine'])[:3,:3]).T
            actual=p(points)
            self.assertLess(np.max(np.linalg.norm((actual-expected)@a[:3,:3].T,axis=1)),1e-4)
        np.testing.assert_allclose(predictor(grid,meta,a,a)(points),points,atol=1e-6)

    def test_reconstruction_checkpoint_and_config(self):
        data=np.arange(125,dtype=np.float32).reshape(5,5,5);c=case(data,np.eye(4))
        arrays,_,_,meta=prepare(c,{'spacing_mm':1.},'cpu')
        torch.manual_seed(42);pair=PairedSiren(3,width=4,layers=1,omega=10.)
        old=endpoint_reconstruction(pair,arrays,32);new=original_reconstruction(pair,c,meta,32)
        self.assertAlmostEqual(old['moving']['mse'],new['moving']['mse'],places=7)
        config=load_config('medical/configs/physical_working_grid.json')
        payload={'model':pair.model_config,'state':pair.state_dict(),'config':config,'working_grid':meta}
        restored=restore_checkpoint(payload);self.assertEqual(restored.working_grid_metadata,meta)
        del payload['working_grid']
        with self.assertRaises(ValueError):restore_checkpoint(payload)
        with self.assertRaises(ValueError):validate_metadata(meta,(4,5,5))
        config['working_grid']['spacing_mm']=2
        with self.assertRaises(ValueError):validate_config(config)

    def test_canonical_sampling_losses_and_gradients(self):
        from medical.registration.objective import TrainingBatch,endpoint_loss
        from medical.models.siren import coordinates_from_indices
        x=np.indices((5,5,5),dtype=np.float32);data=x.sum(0)
        options={'spacing_mm':1.,'reference_geometry':geometry([5]*3,np.eye(4))}
        outputs=[prepare(c,options,'cpu')[0] for c in (case(data,np.eye(4)),case(data[::2,::2,::2],np.diag([2.,2.,2.,1.])))]
        torch.manual_seed(42);pair=PairedSiren(3,width=4,layers=1,omega=10.)
        losses=[];grads=[];coordinates=[]
        for images in outputs:
            indices=torch.randint(images[0].numel(),(32,),generator=torch.Generator().manual_seed(42))
            coords=coordinates_from_indices(indices,images[0].shape);coordinates.append(coords)
            batch=TrainingBatch(coords,images[0].flatten()[indices],images[1].flatten()[indices])
            loss=endpoint_loss(pair,batch,load_config()['objective'],compile_endpoints=False)['total']
            losses.append(loss);grads.append(torch.autograd.grad(loss,tuple(pair.parameters())))
        torch.testing.assert_close(coordinates[0],coordinates[1]);torch.testing.assert_close(*losses)
        for a,b in zip(*grads):torch.testing.assert_close(a,b)

    def test_learn2reg_adapter_routes_original_case(self):
        from medical.abdomen_method import run_case
        from medical.registration.recovery import RecoveryResult
        c=case(np.arange(125,dtype=np.float32).reshape(5,5,5),np.diag([2.,2.,2.,1.]))
        pair=PairedSiren(3,width=4,layers=1,omega=10.)
        grid=identity_grid((5,5,5),'cpu')
        def evaluate(original,prediction,protocol):
            self.assertIs(original,c)
            np.testing.assert_allclose(prediction(np.array([[0,0,0],[4,4,4]])),[[0,0,0],[4,4,4]])
            return {}
        with patch('medical.abdomen_method.validate_case'),patch('medical.abdomen_method.fit_pair',return_value=pair),patch('medical.abdomen_method.recover',return_value=RecoveryResult(grid,grid,[])),patch('medical.abdomen_method.evaluate',side_effect=evaluate):
            result=run_case(c,load_config('medical/configs/physical_working_grid.json'),'cpu')
        self.assertIn('working_grid',result['checkpoint'])
        self.assertIn('original_grid',result['metrics']['reconstruction'])

    def test_registration_loss_gradient_consistency(self):
        from medical.tests import test_siren_derivatives as fixtures
        x=np.indices((9,9,9),dtype=np.float32);data=x[0]+2*x[1]+x[2]
        opts={'spacing_mm':1.,'reference_geometry':geometry([9]*3,np.eye(4))}
        tensors=[prepare(c,opts,'cpu')[0] for c in (case(data,np.eye(4)),case(data[::2,::2,::2],np.diag([2.,2.,2.,1.])))]
        for mode in ('pdlk','inverse_metric'):
            results=[];gradients=[]
            pair=fixtures.make_pair(width=4,omega=10.)
            for images in tensors:
                branch=fixtures.TestLiveRegistration().branch()
                branch.moving,branch.fixed=(v.double() for v in images)
                branch.regularizer=mode
                terms=branch.loss(pair)
                loss=terms['registration']*.01+terms['grid_pdlk' if mode=='pdlk' else 'inverse_metric']*.1
                results.append(loss)
                gradients.append(torch.autograd.grad(loss,tuple(pair.parameters())))
            torch.testing.assert_close(*results)
            for a,b in zip(*gradients):torch.testing.assert_close(a,b)

    def test_tiny_recovery_scratch_success_error_and_metadata_guard(self):
        from pathlib import Path
        from medical.registration.recovery import recover
        c=case(np.arange(125,dtype=np.float32).reshape(5,5,5),np.eye(4))
        _,_,_,meta=prepare(c,{'spacing_mm':1.},'cpu')
        pair=PairedSiren(3,width=4,layers=1,omega=10.)
        pair.requires_working_grid_metadata=True
        settings=load_config()['recovery'];settings['intervals']=2
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError,'metadata'):recover(pair,(5,5,5),settings,root)
            pair.working_grid_metadata=meta
            zero=(torch.zeros(5,5,5,3),torch.zeros(5,5,5))
            with patch('medical.registration.recovery.derivative_grid',return_value=zero):
                result=recover(pair,(5,5,5),settings,root,compile_solver=False)
            torch.testing.assert_close(result.inverse,identity_grid((5,5,5),'cpu'))
            self.assertTrue(all(d['converged'] for d in result.diagnostics))
            self.assertEqual(list(Path(root).iterdir()),[])
            with patch('medical.registration.recovery.derivative_grid',side_effect=RuntimeError('injected')):
                with self.assertRaisesRegex(RuntimeError,'injected'):recover(pair,(5,5,5),settings,root)
            self.assertEqual(list(Path(root).iterdir()),[])

    def test_dirlab_original_landmark_grid_and_masks(self):
        from pathlib import Path
        import nibabel as nib
        from medical.paired_method import run_case
        from medical.registration.recovery import RecoveryResult
        with tempfile.TemporaryDirectory() as root:
            d=Path(root);data=np.arange(125,dtype=np.float32).reshape(5,5,5)
            c=case(data,np.diag([2.,2.,2.,1.]))
            for key,name in (('fixed','T00'),('moving','T50')):
                c[key]['path']=str(d/(name+'.nii.gz'))
                nib.save(nib.Nifti1Image(np.ones((5,5,5),np.uint8),c[key]['affine']),d/(name+'_lung_mask.nii.gz'))
                np.savetxt(d/(name+'_landmarks_voxel.csv'),np.tile([2,2,2],(300,1)),delimiter=',',header='x,y,z',comments='')
            pair=PairedSiren(3,width=4,layers=1,omega=10.)
            grid=identity_grid((5,5,5),'cpu')
            with patch('medical.paired_method.fit_pair',return_value=pair) as fit,patch('medical.paired_method.recover',return_value=RecoveryResult(grid,grid,[])):
                result=run_case(c,load_config('medical/configs/physical_working_grid.json'),'cpu')
            self.assertEqual(tuple(fit.call_args.args[0].shape),(9,9,9))
            self.assertEqual(tuple(fit.call_args.kwargs['moving_mask'].shape),(9,9,9))
            self.assertEqual(result['metrics']['continuous_tre_mm']['mean_mm'],0)
            self.assertEqual(result['metrics']['reverse_T50_to_T00']['continuous_tre_mm']['mean_mm'],0)
            self.assertEqual(result['metrics']['folding_fraction'],0)

    def test_launcher_aggregates_each_psnr_scope(self):
        import io,json
        from contextlib import redirect_stdout
        from pathlib import Path
        from medical.abdomen_experiments import main
        with tempfile.TemporaryDirectory() as root:
            out=Path(root)/'output';config=Path(root)/'config.json';config.write_text('{}')
            job={'method':'ours','pair_index':2,'seed':42,'output':str(out/'case'),
                 'missing_inputs':[],'command':['python','--config',str(config),'--output',str(out/'case')]}
            def run(worker,command,log):
                target=out/'case';target.mkdir()
                scores={key:{'mean':.5} for key in ('DSC','HD95','LogJacDetStd')}
                scores['reconstruction']={scope:{key:{'psnr_db':value} for key in ('moving','fixed')}
                                          for scope,value in (('working_grid',30.),('original_grid',28.))}
                (target/'metrics.json').write_text(json.dumps(scores))
                return {'returncode':0}
            with patch('medical.abdomen_experiments.make_jobs',return_value=[job]),patch('medical.gpu_worker.GPUWorker.run',autospec=True,side_effect=run),redirect_stdout(io.StringIO()):
                main(['--methods','ours','--gpus','0','--output',str(out),'--run'])
            scores=json.loads((out/'summary.json').read_text())['methods']['ours']
            self.assertEqual(scores['working_grid_moving_psnr_db']['mean'],30.)
            self.assertEqual(scores['original_grid_fixed_psnr_db']['mean'],28.)
            self.assertEqual(scores['moving_psnr_db']['scored_pairs'],0)

    def test_absolute_crop_geometry_and_sampling(self):
        from medical.registration.symmetric_registration import SymmetricPairRegistration,sample_crop_indices
        from medical.working_grid import configure_crop
        image=torch.zeros(85,87,89)
        branch=SymmetricPairRegistration(image,image,torch.eye(4),grid_size=8,crop_fraction=.5,
                                        pair_offset_fraction=.2,particles=32,compile_derivatives=False)
        configure_crop(branch,{'crop_extent_mm':80})
        self.assertEqual(branch.crop_shape,(81,81,81))
        self.assertAlmostEqual(branch.pair_radius_mm,2.)
        sample_crop_indices(branch,32,torch.Generator().manual_seed(42))
        torch.testing.assert_close(branch.domain_extent,torch.full((3,),80.))
        positions=branch.sample_positions(torch.Generator().manual_seed(42))
        for p in positions:
            self.assertTrue(((p>=branch.origin+8)&(p<=branch.origin+72)).all())
        with self.assertRaises(ValueError):configure_crop(branch,{'crop_extent_mm':100})
