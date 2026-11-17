"""Native-reduction parity; synthetic CUDA coverage is explicitly opt-in."""
import os,unittest
import torch
from medical.registration.compiled_decoder import three_tap_smoothing,compiled_three_tap_smoothing
from medical.registration.shared_grid import smooth_velocity

class TestDecoder(unittest.TestCase):
 def check(self,device,compiled):
  torch.manual_seed(42)
  field=torch.randn(8,8,8,3,device=device,requires_grad=True);weight=torch.randn_like(field)
  a=smooth_velocity(field,[.75]*3,kernel='three_tap')
  b=(compiled_three_tap_smoothing() if compiled else three_tap_smoothing)(field,(.75,)*3)
  torch.testing.assert_close(a,b)
  ga=torch.autograd.grad((a*weight).sum(),field)[0];gb=torch.autograd.grad((b*weight).sum(),field)[0]
  torch.testing.assert_close(ga,gb)
 def test_eager(self):self.check('cpu',False)
 @unittest.skipUnless(os.environ.get('INR_TEST_CUDA')=='1','Opt-in CUDA checks')
 def test_cuda(self):self.check('cuda',True)

 def test_decoder_semantics(self):
  for integrator in ('rk2',):
   with self.subTest(integrator=integrator):self._decoder_semantics(integrator)

 def _decoder_semantics(self,integrator):
  from medical.registration.symmetric_registration import SymmetricPairRegistration
  from medical.registration.compiled_decoder import box_decoder
  torch.manual_seed(42)
  image=torch.rand(9,9,9)
  branch=SymmetricPairRegistration(image,image.flip(0),torch.eye(4),grid_size=8,particles=64,
      pairs_per_cell=2,compile_velocity=False,crop_fraction=1.,training_integrator=integrator)
  starts=1+torch.rand(64,3)*6
  for magnitude in [.1,100.]:
   fields=tuple((torch.ones(8,8,8,3)*magnitude).requires_grad_() for _ in range(branch.time_strata))
   for reverse in [False,True]:
    ref=branch._direction_loss(starts,fields,reverse)
    source,target=(branch.fixed,branch.moving) if reverse else (branch.moving,branch.fixed)
    actual,finite=box_decoder(starts,fields,source,target,branch.origin,branch.domain_extent,
         branch.extent,8,reverse,integrator=integrator)
    self.assertTrue(bool(finite))
    for key in ref:torch.testing.assert_close(ref[key],actual[key],atol=1e-6,rtol=1e-5)
    self.assertEqual(actual['valid_particles'],64)
    self.assertEqual(actual['skipped'],0)
    ga=torch.autograd.grad(ref['similarity']+ref['boundary'],fields,retain_graph=True)
    gb=torch.autograd.grad(actual['similarity']+actual['boundary'],fields,retain_graph=True)
    for x,y in zip(ga,gb):
     torch.testing.assert_close(x,y,rtol=1e-4,atol=1e-6)
     self.assertTrue(torch.isfinite(x).all());self.assertGreater(float(x.abs().sum()),0)

 @unittest.skipUnless(os.environ.get('INR_TEST_CUDA')=='1','Opt-in CUDA checks')
 def test_full_decoder_cuda_gradients(self):
  for integrator in ('rk2',):
   with self.subTest(integrator=integrator):self._full_decoder_cuda_gradients(integrator)

 def _full_decoder_cuda_gradients(self,integrator):
  from medical.registration.compiled_decoder import box_decoder,compiled_box_decoder
  torch.manual_seed(42)
  source=torch.rand(17,17,17,device='cuda');target=torch.rand_like(source)
  origin=torch.ones(3,device='cuda')*2;domain=torch.ones(3,device='cuda')*8;extent=torch.ones(3,device='cuda')*16
  # Populate the 8^3 GW-NCC bins; singleton windows have ill-conditioned
  # FP32 variance subtraction and do not represent the 64k training sampler.
  starts=origin+.1*domain+torch.rand(4096,3,device='cuda')*.8*domain
  fields=tuple((torch.ones(4,4,4,3,device='cuda')*4).requires_grad_() for _ in range(2))
  for reverse in [False,True]:
   args=(starts,fields,source,target,origin,domain,extent,4,reverse)
   a,_=box_decoder(*args,integrator=integrator);b,finite=compiled_box_decoder()(*args,integrator=integrator)
   self.assertTrue(finite)
   for key in a:torch.testing.assert_close(a[key],b[key],rtol=1e-4,atol=1e-5)
   ga=torch.autograd.grad(a['similarity']+a['boundary'],fields)
   gb=torch.autograd.grad(b['similarity']+b['boundary'],fields)
   for x,y in zip(ga,gb):torch.testing.assert_close(x,y,rtol=1e-3,atol=1e-5)
