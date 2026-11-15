"""Synthetic physical velocity-kernel geometry and gradients; no optimizer."""
import copy,unittest
import numpy as np
import torch
from medical.configuration import load_config,validate_config
from medical.registration.symmetric_registration import SymmetricPairRegistration
from medical.registration.shared_grid import smooth_velocity

class TestMetricSmoothing(unittest.TestCase):
 def branch(self,shape,spacing,sigma=None):
  image=torch.zeros(shape,dtype=torch.float64);affine=np.diag([*spacing,1.])
  affine[:3,:3]=np.array([[0,-1,0],[1,0,0],[0,0,-1]])@affine[:3,:3]
  options={} if sigma is None else {'sigma_mm':sigma}
  return SymmetricPairRegistration(image,image,affine,grid_size=8,particles=8,
      gaussian_kernel='three_tap',sigma_cells=.75,compile_velocity=False,**options)
 def test_axes_kernel_distances_and_gradients(self):
  b=self.branch((17,25,33),(2.5,1.25,.8),7.5)
  widths=torch.tensor([16*2.5/8,24*1.25/8,32*.8/8],dtype=torch.float64)
  torch.testing.assert_close(torch.tensor(b.sigma,dtype=torch.float64),7.5/widths)
  impulse=torch.zeros(8,8,8,3,dtype=torch.float64);impulse[4,4,4,0]=1;impulse.requires_grad_()
  out=smooth_velocity(impulse,b.sigma,kernel='three_tap')
  for axis in range(3):
   index=[4,4,4,0];index[axis]+=1
   torch.testing.assert_close(out[tuple(index)]/out[4,4,4,0],torch.exp(-.5*(widths[axis]/7.5)**2))
  gradient=torch.autograd.grad(out.square().sum(),impulse)[0]
  self.assertTrue(torch.isfinite(gradient).all());self.assertGreater(float(gradient.abs().sum()),0)
 def test_equal_physical_crops_and_resize(self):
  a=self.branch((17,17,17),(1,1,1),3.)
  b=self.branch((9,17,33),(2,1,.5),3.)
  self.assertEqual(a.sigma,b.sigma)
  b.set_box(torch.zeros(3),(5,9,17))
  torch.testing.assert_close(torch.tensor(b.sigma),torch.tensor(a.sigma)*2)
 def test_default_and_config_validation(self):
  self.assertEqual(self.branch((17,25,33),(2.5,1.25,.8)).sigma,[.75]*3)
  c=load_config('medical/configs/native_physical_units_sigma7p5mm_endpoint_ncc_32k.json')
  for value in [0,-1,float('nan'),True,None]:
   bad=copy.deepcopy(c);bad['objective']['registration']['sigma_mm']=value
   with self.assertRaises(ValueError):validate_config(bad)
