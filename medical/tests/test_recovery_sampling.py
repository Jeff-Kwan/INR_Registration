"""Fixed recovery geometry, native units and exception cleanup; synthetic only."""
from pathlib import Path
import tempfile
import unittest
import torch
from medical.registration.recovery import recover
from medical.registration.geometry import identity_grid


class LinearTranslation(torch.nn.Module):
    def __init__(self,shape,fail=False):
        super().__init__();self.reference=torch.nn.Parameter(torch.zeros(()))
        self.shape=shape;self.fail=fail
        self.gradient=torch.tensor([1.,2.,3.]);self.translation=torch.tensor([.2,.4,.6])

    def prepare(self,**unused):return self

    def at(self,time):return self

    def derivatives(self,coordinates,scale,**unused):
        if self.fail:raise RuntimeError('synthetic derivative failure')
        extent=coordinates.new_tensor([n-1 for n in reversed(self.shape)])
        coefficients=self.gradient*extent/2
        g=(coefficients*scale).expand_as(coordinates)
        q=-(self.gradient*self.translation).sum().expand(coordinates.shape[0])
        return g,q


class TestRecoverySampling(unittest.TestCase):
    settings={'intervals':2,'smoothness':.01,'ridge':1.,'iterations':100,'tolerance':1e-6,'substeps':2}

    def test_native_units_and_original_grid_return(self):
        shape=(13,15,17);pair=LinearTranslation(shape).eval().requires_grad_(False)
        with tempfile.TemporaryDirectory(prefix='test-native-recovery-') as scratch:
            full=recover(pair,shape,self.settings,scratch,compile_derivatives=False,compile_solver=False)
            same=recover(pair,shape,self.settings,scratch,max_axis=17,compile_derivatives=False,compile_solver=False)
            coarse=recover(pair,shape,self.settings,scratch,max_axis=9,compile_derivatives=False,compile_solver=False)
            self.assertEqual(list(Path(scratch).iterdir()),[])
        torch.testing.assert_close(full.forward,same.forward,rtol=0,atol=0)
        torch.testing.assert_close(full.inverse,same.inverse,rtol=0,atol=0)
        torch.testing.assert_close(full.forward,coarse.forward,rtol=1e-5,atol=2e-5)
        torch.testing.assert_close(full.inverse,coarse.inverse,rtol=1e-5,atol=2e-5)
        identity=identity_grid(shape,'cpu')
        expected=pair.translation*(14/15)
        torch.testing.assert_close(coarse.forward,identity+expected,rtol=1e-5,atol=2e-5)
        torch.testing.assert_close(coarse.inverse,identity-expected,rtol=1e-5,atol=2e-5)
        self.assertEqual(tuple(coarse.forward.shape),(*shape,3))
        self.assertTrue(all(d['converged'] for d in coarse.diagnostics))
        self.assertTrue(all(d['velocity_units']=='original_native_voxels' for d in coarse.diagnostics))

    def test_failure_cleans_scratch_and_physical_models_are_rejected(self):
        shape=(13,15,17);pair=LinearTranslation(shape,fail=True)
        with tempfile.TemporaryDirectory(prefix='test-native-recovery-error-') as scratch:
            with self.assertRaisesRegex(RuntimeError,'synthetic derivative failure'):
                recover(pair,shape,self.settings,scratch,max_axis=9,compile_derivatives=False,compile_solver=False)
            self.assertEqual(list(Path(scratch).iterdir()),[])
            pair.requires_physical_units_metadata=True
            with self.assertRaises(ValueError):
                recover(pair,shape,self.settings,scratch,max_axis=9,compile_derivatives=False,compile_solver=False)
            self.assertEqual(list(Path(scratch).iterdir()),[])
