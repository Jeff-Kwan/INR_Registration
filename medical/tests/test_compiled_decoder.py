"""Native-reduction parity; synthetic CUDA coverage is explicitly opt-in."""
import os,unittest
import torch
from medical.registration.compiled_decoder import box_statistics,three_tap_smoothing,compiled_box_statistics,compiled_three_tap_smoothing
from medical.registration.shared_grid import smooth_velocity
from medical.registration.ncc import correlation

class TestDecoder(unittest.TestCase):
 def check(self,device,compiled):
  torch.manual_seed(42)
  for count in [2,31,32,257]:
   a=torch.randn(count,device=device,requires_grad=True);b=torch.randn_like(a,requires_grad=True)
   fn=compiled_box_statistics() if compiled else box_statistics
   mse,ncc=fn(a,b)
   expected=(a-b).square().mean()+1-correlation(a,b,1e-8)
   torch.testing.assert_close(mse+ncc,expected,rtol=1e-5,atol=1e-6)
   ga=torch.autograd.grad(mse+ncc,(a,b));gb=torch.autograd.grad(expected,(a,b))
   for x,y in zip(ga,gb):torch.testing.assert_close(x,y,rtol=1e-4,atol=1e-5)
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
  from medical.registration.symmetric_registration import SymmetricPairRegistration
  from medical.registration.compiled_decoder import box_decoder
  torch.manual_seed(42)
  image=torch.rand(9,9,9)
  branch=SymmetricPairRegistration(image,image.flip(0),torch.eye(4),grid_size=8,particles=64,
      pairs_per_cell=2,compile_velocity=False,crop_fraction=1.)
  starts=1+torch.rand(64,3)*6
  for magnitude in [.1,100.]:
   fields=tuple((torch.ones(8,8,8,3)*magnitude).requires_grad_() for _ in range(2))
   for reverse in [False,True]:
    ref=branch._direction_loss(starts,fields,reverse)
    source,target=(branch.fixed,branch.moving) if reverse else (branch.moving,branch.fixed)
    actual,finite=box_decoder(starts,fields,source,target,branch.origin,branch.domain_extent,
         branch.extent,8,reverse)
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
  from medical.registration.compiled_decoder import box_decoder,compiled_box_decoder
  torch.manual_seed(42)
  source=torch.rand(17,17,17,device='cuda');target=torch.rand_like(source)
  starts=torch.rand(32,3,device='cuda')*2+7
  origin=torch.ones(3,device='cuda')*2;domain=torch.ones(3,device='cuda')*8;extent=torch.ones(3,device='cuda')*16
  fields=tuple((torch.ones(4,4,4,3,device='cuda')*4).requires_grad_() for _ in range(2))
  for reverse in [False,True]:
   args=(starts,fields,source,target,origin,domain,extent,4,reverse)
   a,_=box_decoder(*args);b,finite=compiled_box_decoder()(*args)
   self.assertTrue(finite)
   for key in a:torch.testing.assert_close(a[key],b[key],rtol=1e-4,atol=1e-5)
   ga=torch.autograd.grad(a['similarity']+a['boundary'],fields)
   gb=torch.autograd.grad(b['similarity']+b['boundary'],fields)
   for x,y in zip(ga,gb):torch.testing.assert_close(x,y,rtol=1e-3,atol=1e-5)

 @unittest.skipUnless(os.environ.get('INR_TEST_CUDA')=='1','Opt-in CUDA checks')
 def test_native_graph_reuses_fresh_origins_fields_and_particles(self):
  from medical.registration.symmetric_registration import SymmetricPairRegistration
  torch.manual_seed(42)
  image = torch.randn(17, 19, 23, device='cuda')
  branch = SymmetricPairRegistration(image, image.flip(0), torch.eye(4), grid_size=4,
      particles=256, crop_fraction=.5, compile_velocity=True, gaussian_kernel='three_tap')
  branch.enable_training_decoder_graph()
  for origin, magnitude in (([1, 1, 1], .1), ([3, 4, 5], 100.)):
   branch.set_origin(image.new_tensor(origin))
   fields = tuple((torch.randn(4, 4, 4, 3, device='cuda') * magnitude).requires_grad_()
                  for _ in range(2))
   starts = branch.sample_positions()
   a, b = [branch._direction_loss(p, fields, reverse) for p, reverse in zip(starts, [False, True])]
   reference = .5 * (a['similarity'] + b['similarity']) + .5 * (a['boundary'] + b['boundary'])
   actual, finite = branch._training_decoder(*fields, *starts, branch.origin)
   self.assertTrue(bool(finite))
   torch.testing.assert_close(actual, reference, rtol=0, atol=0)
   for got, expected in zip(torch.autograd.grad(actual, fields), torch.autograd.grad(reference, fields)):
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-7)
  bad_field = torch.full_like(fields[0], float('nan'), requires_grad=True)
  _, finite = branch._training_decoder(bad_field, fields[1], *starts, branch.origin)
  self.assertFalse(bool(finite))
