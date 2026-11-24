"""Synthetic checks for crop exits and failed registration solves."""
import unittest
from unittest.mock import patch
import torch
from medical.registration.symmetric_registration import SymmetricPairRegistration

class TestRegistrationGuards(unittest.TestCase):
 def branch(self):
  image=torch.rand(25,25,25)
  return SymmetricPairRegistration(image,image,torch.eye(4),grid_size=4,particles=64,
      crop_fraction=.36,compile_derivatives=False,compile_velocity=False)
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
    for options in ({}, {'deferred_checks':[], 'need_local_transport':False},
                    {'deferred_checks':[], 'need_registration':False}):
     with self.assertRaises(torch.linalg.LinAlgError):
      branch.loss(pair,diagnostics=False,**options)
    # Bypass transport after the injected fault so the test observes only the
    # LU dispatch and the full-grid energy consumed by the trainer's guard.
    direction={'similarity':image.new_zeros(()),'boundary':image.new_zeros(())}
    with patch.object(branch,'_direction_loss',return_value=direction):
     terms=branch.loss(pair,diagnostics=False,deferred_checks=[])
     self.assertFalse(bool(torch.isfinite(terms['local_transport_loss'])))
