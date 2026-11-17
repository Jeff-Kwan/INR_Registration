"""Synthetic crop-only mask semantics; no optimizer updates or recovery."""
import copy
import unittest
from unittest.mock import patch
import torch
from medical.registration.symmetric_registration import SymmetricPairRegistration, sample_crop_indices

class TestMaskedCrop(unittest.TestCase):
 def branch(self):
  image=torch.rand(25,25,25)
  return SymmetricPairRegistration(image,image,torch.eye(4),grid_size=4,particles=64,
      crop_fraction=.36,compile_derivatives=False,compile_velocity=False)
 def test_disjoint_masks_center_without_rejection(self):
  b=self.branch();b.crop_center_sites=[torch.tensor([[5.,5.,5.]]),torch.tensor([[19.,19.,19.]])]
  generator=torch.Generator().manual_seed(42);seen=set()
  for _ in range(30):
   ids=sample_crop_indices(b,128,generator)
   center=b.origin+b.domain_extent/2
   seen.add(tuple(center.tolist()))
   xyz=torch.stack((ids%25,(ids//25)%25,ids//625),-1)
   self.assertTrue(b.supported(xyz).all())
  self.assertEqual(seen,{(5.,5.,5.),(19.,19.,19.)})
 def test_edge_center_shifts_box_and_particles_cover_interior(self):
  b=self.branch();b.crop_center_sites=[torch.zeros(1,3)]
  sample_crop_indices(b,128,torch.Generator().manual_seed(42))
  torch.testing.assert_close(b.origin,torch.zeros(3))
  a,c=b.sample_positions()
  self.assertTrue(b.supported(a).all());self.assertFalse(torch.equal(a,c))
  self.assertGreater(len(a.unique(dim=0)),60)

 def test_crop_exit_keeps_ncc_gradient_inside_image(self):
  b=self.branch();b.set_origin(torch.tensor([4.,4.,4.]))
  starts=torch.tensor([[11.,6.,6.],[11.,7.,7.],[11.,8.,8.],[11.,9.,9.]])
  fields=[]
  for _ in range(2):
   f=torch.zeros(4,4,4,3);f[...,0]=4;fields.append(f.requires_grad_())
  terms=b._direction_loss(starts,tuple(fields),False)
  self.assertEqual(float(terms['outside']),1.)
  self.assertEqual(float(terms['valid_particles']),4.)
  self.assertEqual(float(terms['skipped']),0.)
  gradients=torch.autograd.grad(terms['similarity'],fields)
  self.assertTrue(all(torch.isfinite(g).all() and g.abs().sum()>0 for g in gradients))

 @unittest.skipUnless(torch.cuda.is_available(), 'CUDA sampling')
 def test_cuda_mask_centers_preserve_native_draws(self):
  image=torch.empty((19,21,25),device='cuda')
  b=SymmetricPairRegistration(image,image,torch.eye(4),grid_size=4,particles=64,
      crop_fraction=.36,compile_derivatives=False,compile_velocity=False)
  b.crop_center_sites=[torch.arange(9,device='cuda').reshape(3,3).float(),
                       torch.arange(15,device='cuda').reshape(5,3).float()]
  expected=torch.Generator(device='cuda').manual_seed(42)
  actual=torch.Generator(device='cuda').manual_seed(42)
  for _ in range(24):
   endpoint=int(torch.randint(2,(),device='cuda',generator=expected))
   sites=b.crop_center_sites[endpoint]
   center=sites[torch.randint(len(sites),(),device='cuda',generator=expected)]
   origin=torch.minimum((center-b.crop_half).round().long().clamp_min(0),b.crop_ranges-1)
   nz,ny,nx=b.crop_shape
   local=torch.randint(nz*ny*nx,(16,),device='cuda',generator=expected)
   indices=((local//(ny*nx)+origin[0])*image.shape[1]+local//nx%ny+origin[1])*image.shape[2]+local%nx+origin[2]
   self.assertTrue(torch.equal(sample_crop_indices(b,16,actual),indices))
   self.assertTrue(torch.equal(b.origin,origin.flip(0).float()))
  self.assertEqual(actual.get_offset(),expected.get_offset())



 def test_singular_solve_dispatch(self):
  from medical.models.siren import PairedSiren
  from medical.registration.symmetric_registration import _pair_velocity_chunk
  for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
   image=torch.zeros((8,8,8),device=device)
   pair=PairedSiren(width=4,layers=1,omega=30.).to(device)
   with torch.no_grad():
    for parameter in pair.parameters():parameter.zero_()
   branch=SymmetricPairRegistration(image,image,torch.eye(4),grid_size=2,particles=8,
       compile_velocity=True,compile_derivatives=False,gaussian_kernel='three_tap')
   branch.eta=0.  # Fault injection: every cell's zero matrix is singular.
   branch.resample_queries()
   compiled=torch.compile(_pair_velocity_chunk,backend='eager',fullgraph=True)
   with patch('medical.registration.symmetric_registration._compiled_pair_velocity_chunk',return_value=compiled):
    with self.assertRaises(torch.linalg.LinAlgError):
     branch._field_terms(pair,image.new_tensor(.5),return_energy=True)
    for options in ({}, {'deferred_checks':[], 'need_pdlk':False},
                    {'deferred_checks':[], 'need_registration':False}):
     with self.assertRaises(torch.linalg.LinAlgError):
      branch.loss(pair,diagnostics=False,**options)
    # Bypass transport after the injected fault so the test observes only the
    # LU dispatch and the full-grid energy consumed by the trainer's guard.
    direction={'similarity':image.new_zeros(()),'boundary':image.new_zeros(())}
    with patch.object(branch,'_direction_loss',return_value=direction):
     terms=branch.loss(pair,diagnostics=False,deferred_checks=[])
     self.assertFalse(bool(torch.isfinite(terms['grid_pdlk'])))
