"""Analytical loss/gradient checks only; no optimizer steps."""
import ast
import unittest
import torch
import torch.nn.functional as F
from baselines.sources import upstream_imports,git


class ThinWindowTests(unittest.TestCase):
    def test_unchanged_supported_volume_loss_and_gradients(self):
        with upstream_imports('nir') as root:
            import loss_functions
            tree=ast.parse(git(root,'show','HEAD:src/loss_functions.py'))
            node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='ncc')
            namespace=dict(torch=torch,F=F)
            exec(compile(ast.Module(body=[node],type_ignores=[]),'native_ncc','exec'),namespace)
            torch.manual_seed(42)
            a=torch.rand(1,12,13,14,1)*.5;b=torch.rand_like(a)*.5
            a[:,:9,:9,:9]=0;b[:,:9,:9,:9]=0
            x=(2*a-1).requires_grad_(True);truth=dict(img=2*b-1)
            torch.manual_seed(42);expected=namespace['ncc'](dict(model_out=x),truth)['ncc']
            torch.manual_seed(42);actual=loss_functions.ncc(dict(model_out=x),truth)['ncc']
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            ga=torch.autograd.grad(actual,x,retain_graph=True)[0];gb=torch.autograd.grad(expected,x)[0]
            torch.testing.assert_close(ga,gb,rtol=0,atol=0)

    def test_thin_depths_are_finite_with_correct_kernel(self):
        from unittest.mock import patch
        with upstream_imports('nir'):
            import loss_functions
            for depth in (7,8):
                torch.manual_seed(42)
                x=(torch.rand(1,13,14,depth,1)*.8-1).requires_grad_(True)
                y=torch.rand_like(x)*.8-1
                original=F.conv3d;shapes=[]
                def record(a,k,*args,**kwargs):
                    shapes.append(tuple(k.shape[-3:]));return original(a,k,*args,**kwargs)
                with patch.object(F,'conv3d',side_effect=record):
                    value=loss_functions.ncc(dict(model_out=x),dict(img=y))['ncc']
                self.assertEqual(set(shapes),{(9,9,7)})
                self.assertTrue(torch.isfinite(value));self.assertTrue(torch.isfinite(torch.autograd.grad(value,x)[0]).all())
