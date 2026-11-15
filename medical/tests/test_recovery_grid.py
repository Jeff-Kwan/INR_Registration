"""Independent recovery resolution: synthetic systems, no optimizer/real cases."""
import itertools
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from medical.configuration import load_config
from medical.models.siren import PairedSiren
from medical.registration.flow import estimate_velocity_from_derivatives as solve,laplacian,stencil_laplacian
from medical.registration.recovery import derivative_grid,recover
from medical.working_grid import geometry,recovery_geometry,prepare,predictor


class TestRecoveryGrid(unittest.TestCase):
    def test_geometry_preserves_domain(self):
        a=np.eye(4);a[:3,:3]=[[0,-1,0],[1,0,0],[0,0,-1]];a[:3,3]=[5,7,9]
        working=geometry([8,10,12],a)
        grid=recovery_geometry(working,2.)
        self.assertEqual(grid['shape'],[5,6,7])
        b=np.asarray(grid['affine'])
        np.testing.assert_allclose(b[:3,3],a[:3,3])
        np.testing.assert_allclose(b[:3,:3]@(np.array(grid['shape'])-1),a[:3,:3]@np.array([7,9,11]))
        self.assertEqual(recovery_geometry(working),working)

    def test_default_original_anisotropic_spacing_and_frozen_reference(self):
        from medical.working_grid import validate_metadata, restore_checkpoint
        affine=np.eye(4);affine[:3,:3]=[[0,-2,0],[.5,0,0],[0,0,-3]]
        data=np.arange(5*4*3,dtype=np.float32).reshape(5,4,3)
        case={k:{'data':data,'affine':affine} for k in ('moving','fixed')}
        arrays,_,_,metadata=prepare(case,{'spacing_mm':1.},'cpu')
        self.assertEqual(metadata['recovery_spacing_mm'],'original')
        self.assertEqual(metadata['recovery_geometry']['shape'],list(data.shape))
        np.testing.assert_allclose(metadata['recovery_geometry']['affine'],affine)
        validate_metadata(metadata,arrays[0].shape)
        pair=PairedSiren(3,width=4,layers=1)
        restored=restore_checkpoint({'config':{'working_grid':{'spacing_mm':1.}},
            'model':pair.model_config,'state':pair.state_dict(),'working_grid':metadata})
        self.assertEqual(restored.working_grid_metadata,metadata)
        # Supplied frozen reference determines spacing, even for differently sampled inputs.
        other=np.arange(3*7*7,dtype=np.float32).reshape(3,7,7)
        other_affine=affine.copy();other_affine[:3,:3]=affine[:3,:3]@np.diag([2,.5,1/3])
        resampled={k:{'data':other,'affine':other_affine} for k in ('moving','fixed')}
        _,_,_,again=prepare(resampled,{'spacing_mm':1.,'reference_geometry':metadata['reference_geometry']},'cpu')
        self.assertEqual(again['recovery_geometry'],metadata['recovery_geometry'])
        _,_,_,explicit=prepare(case,{'spacing_mm':1.,'recovery_spacing_mm':1.},'cpu')
        self.assertEqual(explicit['recovery_geometry'],explicit['working_geometry'])

    def test_weighted_operator_direct_solution(self):
        torch.manual_seed(42)
        for shape,spacing in (((3,4),(1.5,2.)),((2,3,4),(2.,1.25,3.))):
            d=len(shape);g=torch.randn(*shape,d,dtype=torch.float64);q=torch.randn(*shape,dtype=torch.float64)
            points=list(itertools.product(*(range(n) for n in shape)));ids={p:i for i,p in enumerate(points)}
            matrix=torch.eye(g.numel(),dtype=g.dtype)*.05
            for p,i in ids.items():
                matrix[i*d:(i+1)*d,i*d:(i+1)*d]+=torch.outer(g[p],g[p])
                for axis,h in enumerate(spacing):
                    neighbor=list(p);neighbor[axis]+=1;neighbor=tuple(neighbor)
                    if neighbor not in ids:continue
                    for c in range(d):
                        x,y=i*d+c,ids[neighbor]*d+c;w=.1/h**2
                        matrix[x,x]+=w;matrix[y,y]+=w;matrix[x,y]-=w;matrix[y,x]-=w
            rhs=(-g*q[...,None]).flatten();expected=torch.linalg.solve(matrix,rhs).reshape_as(g)
            actual,diag=solve(g,q,smoothness=.1,ridge=.05,spacing=spacing,tolerance=1e-10)
            torch.testing.assert_close(actual,expected,rtol=1e-7,atol=1e-8)
            self.assertTrue(diag['converged'])
            self.assertLess(float((rhs-matrix@actual.flatten()).norm()/rhs.norm()),1e-10)
            weights=g.new_tensor(spacing).square().reciprocal()
            torch.testing.assert_close(laplacian(g,weights),stencil_laplacian(g,weights))

    def test_derivative_units_and_translation_integration(self):
        torch.manual_seed(42);pair=PairedSiren(3,width=4,layers=1,omega=10.)
        fine,q=derivative_grid(pair,(5,5,5),.3,compile_derivatives=False)
        coarse,r=derivative_grid(pair,(3,3,3),.3,compile_derivatives=False)
        torch.testing.assert_close(coarse/2,fine[::2,::2,::2])
        torch.testing.assert_close(r,q[::2,::2,::2])
        data=np.arange(125,dtype=np.float32).reshape(5,5,5)
        case={k:{'data':data,'affine':np.eye(4)} for k in ('moving','fixed')}
        arrays,_,_,metadata=prepare(case,{'spacing_mm':1.,'recovery_spacing_mm':2.},'cpu')
        self.assertEqual(tuple(arrays[0].shape),(5,5,5))
        pair.working_grid_metadata=metadata
        settings=load_config()['recovery'];settings['intervals']=2
        def derivatives(pair,shape,*args):
            self.assertEqual(shape,(3,3,3))
            return torch.full((*shape,3),2.),torch.zeros(shape)
        def solver(g,q,**options):
            torch.testing.assert_close(g,torch.ones_like(g))
            self.assertEqual(options['spacing'],(2.,2.,2.))
            v=torch.zeros_like(g);v[...,0]=2.
            return v,{'converged':True}
        with tempfile.TemporaryDirectory() as root,patch('medical.registration.recovery.derivative_grid',side_effect=derivatives):
            result=recover(pair,(5,5,5),settings,root,velocity_solver=solver)
            self.assertEqual(list(Path(root).iterdir()),[])
        points=np.array([[0,0,0],[2,2,2],[4,4,4]],np.float32)
        forward=predictor(result.forward,metadata,np.eye(4),np.eye(4))(points)
        reverse=predictor(result.inverse,metadata,np.eye(4),np.eye(4))(points)
        np.testing.assert_allclose(forward,points+[2,0,0],atol=1e-5)
        np.testing.assert_allclose(reverse,points-[2,0,0],atol=1e-5)

    @unittest.skipUnless(os.environ.get('INR_TEST_CUDA')=='1','Opt-in synthetic compiled CUDA check')
    def test_weighted_compiled_parity(self):
        torch.manual_seed(42);g=torch.randn(5,6,7,3,device='cuda')*.1;q=torch.randn(5,6,7,device='cuda')
        a,_=solve(g,q,spacing=(2.,1.5,3.),iterations=40,tolerance=1e-12,compile_solver=False)
        b,_=solve(g,q,spacing=(2.,1.5,3.),iterations=40,tolerance=1e-12,compile_solver=True)
        self.assertLess(float((a-b).norm()/a.norm()),1e-3)
