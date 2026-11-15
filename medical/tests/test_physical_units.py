"""Synthetic unit conversion and recovery routing; no fitting or saved-model runs."""
import copy,tempfile,unittest,os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from medical.configuration import load_config,validate_config
from medical.models.siren import PairedSiren
from medical.physical_units import metadata,validate_metadata,restore_checkpoint
from medical.registration.symmetric_registration import SymmetricPairRegistration,shared_pair_velocity,shared_pair_energy
from medical.registration.recovery import recover
from medical.registration.geometry import identity_grid

class TestPhysicalUnits(unittest.TestCase):
 def setup_pair(self, spacing=(2.5,1.2,.8), mode='mm'):
  torch.manual_seed(42);image=torch.rand(7,9,11,dtype=torch.float64)
  affine=np.diag([*spacing,1.]);affine[:3,:3]=np.array([[0,-1,0],[1,0,0],[0,0,-1]])@affine[:3,:3]
  b=SymmetricPairRegistration(image,image,affine,grid_size=2,particles=8,pairs_per_cell=2,
      compile_derivatives=False,compile_velocity=False,spatial_units=mode,gaussian_kernel='three_tap')
  b.resample_queries()
  pair=PairedSiren(3,width=4,layers=2,omega=3.).double()
  with torch.no_grad():
   for parameter in pair.target.parameters():parameter.add_(torch.randn_like(parameter)*.01)
  return pair,b,affine
 def test_analytical_scaling_solve_energy_and_live_gradients(self):
  pair,b,_=self.setup_pair();coords=b.coordinates.flatten(0,2);prepared=pair.prepare().at(.37)
  scale=coords.new_tensor([2/10,2/8,2/6]);g,q,h,m=prepared.derivatives(coords,scale,differential=True,compile_derivatives=False)
  gm,qm,hm,mm=prepared.derivatives(coords,b.spatial_scale,differential=True,compile_derivatives=False)
  spacing=b.spacing_xyz
  for actual,expected in [(gm,g/spacing),(qm,q),(hm,h/spacing[:,None]/spacing[None,:]),(mm,m/spacing)]:
   torch.testing.assert_close(actual,expected)
  gm=gm.reshape(-1,4,3);qm=qm.reshape(-1,4);hm=hm.reshape(-1,4,3,3);mm=mm.reshape(-1,4,3)
  v=shared_pair_velocity(gm,qm,hm,mm,b.c,b.eta)
  from medical.registration.shared_grid import smooth_velocity
  field=smooth_velocity((v/spacing).reshape(2,2,2,3),b.sigma,kernel=b.gaussian_kernel)
  energy=shared_pair_energy(gm,qm,hm,mm,v,b.c,b.eta).mean()
  actual_field,actual_energy=b._field_terms(pair,.37,return_energy=True)
  torch.testing.assert_close(actual_field,field);torch.testing.assert_close(actual_energy,energy)
  ga=torch.autograd.grad(actual_field.square().mean()+actual_energy,tuple(pair.parameters()))
  gb=torch.autograd.grad(field.square().mean()+energy,tuple(pair.parameters()))
  for a,c in zip(ga,gb):torch.testing.assert_close(a,c);self.assertTrue(torch.isfinite(a).all())
  self.assertTrue(all(float(x.abs().sum())>0 for x in ga))
 def test_one_mm_regression(self):
  p,a,_=self.setup_pair((1.,1.,1.));_,b,_=self.setup_pair((1.,1.,1.),'voxel')
  for x,y in zip(a._field_terms(p,.4,return_energy=True),b._field_terms(p,.4,return_energy=True)):torch.testing.assert_close(x,y,rtol=0,atol=0)
 def test_native_recovery_spacing_conversion_cleanup(self):
  pair,b,affine=self.setup_pair();pair.physical_units_metadata=metadata((7,9,11),affine)
  pair.requires_physical_units_metadata=True;settings=load_config()['recovery'];settings['intervals']=2
  h=b.spacing_xyz.float()
  def derivatives(pair,shape,*args):return h.expand(*shape,3).clone(),torch.zeros(shape)
  def solver(g,q,**options):
   torch.testing.assert_close(g,torch.ones_like(g));self.assertEqual(options['spacing'],(2.5,1.2,.8))
   return g*h,{'converged':True}
  pair=pair.float()
  with tempfile.TemporaryDirectory() as root,patch('medical.registration.recovery.derivative_grid',side_effect=derivatives):
   result=recover(pair,(7,9,11),settings,root,velocity_solver=solver)
   grid=identity_grid((7,9,11),'cpu');torch.testing.assert_close(result.forward,grid+1);torch.testing.assert_close(result.inverse,grid-1)
   self.assertEqual(list(Path(root).iterdir()),[])
   with patch('medical.registration.recovery.derivative_grid',side_effect=RuntimeError('injected')):
    with self.assertRaisesRegex(RuntimeError,'injected'):recover(pair,(7,9,11),settings,root)
   self.assertEqual(list(Path(root).iterdir()),[])
  del pair.physical_units_metadata
  with tempfile.TemporaryDirectory() as root:
   with self.assertRaisesRegex(ValueError,'checkpoint geometry'):recover(pair,(7,9,11),settings,root)
 def test_checkpoint_and_config_guards(self):
  config=load_config('medical/configs/native_physical_units_endpoint_ncc_32k.json')
  pair,b,affine=self.setup_pair();pair=pair.float();value=metadata((7,9,11),affine)
  payload={'config':config,'model':pair.model_config,'state':pair.state_dict(),'physical_units':value}
  restored=restore_checkpoint(payload);self.assertEqual(restored.physical_units_metadata,value)
  for key in ['physical_units']:
   bad=copy.deepcopy(payload);del bad[key]
   with self.assertRaises(ValueError):restore_checkpoint(bad)
  bad=copy.deepcopy(value);bad['spacing_xyz_mm'][0]*=2
  with self.assertRaises(ValueError):validate_metadata(bad,(7,9,11))
  for shape in [(8,9,11),(7,9)]:
   with self.assertRaises(ValueError):validate_metadata(value,shape)
  bad=copy.deepcopy(config);bad['working_grid']={'spacing_mm':1.}
  with self.assertRaises(ValueError):validate_config(bad)
  affine[0,0]+=.3
  with self.assertRaises(ValueError):metadata((7,9,11),affine)

 def test_artifact_roundtrip_and_metadata_required_before_update(self):
  from medical.artifacts import save_checkpoint,load_checkpoint
  from medical.registration.training import fit_pair
  config=load_config('medical/configs/native_physical_units_endpoint_ncc_32k.json')
  pair,b,affine=self.setup_pair();pair=pair.float();config['model']=pair.model_config
  pair.physical_units_metadata=metadata((7,9,11),affine)
  with tempfile.TemporaryDirectory() as root:
   path=Path(root)/'checkpoint.pt';save_checkpoint(path,pair,config,{})
   restored,payload=load_checkpoint(path)
   self.assertEqual(restored.physical_units_metadata,pair.physical_units_metadata)
   del pair.physical_units_metadata
   with self.assertRaises(ValueError):save_checkpoint(path,pair,config,{})
  with self.assertRaises(ValueError):fit_pair(torch.zeros(3,3,3),torch.zeros(3,3,3),config)

 @unittest.skipUnless(os.environ.get('INR_TEST_CUDA')=='1','Opt-in synthetic CUDA')
 def test_compiled_physical_field_gradients(self):
  pair,b,affine=self.setup_pair();pair=pair.float().cuda();image=b.moving.float().cuda()
  branches=[SymmetricPairRegistration(image,image,affine,grid_size=2,particles=8,pairs_per_cell=2,
    gaussian_kernel='three_tap',spatial_units='mm',compile_derivatives=False,compile_velocity=compiled) for compiled in (False,True)]
  outputs=[];gradients=[]
  for branch in branches:
   branch.coordinates=b.coordinates.float().cuda()
   field,energy=branch._field_terms(pair,.37,return_energy=True)
   outputs.append((field,energy));gradients.append(torch.autograd.grad(field.square().mean()+energy,tuple(pair.parameters())))
  for a,c in zip(*outputs):torch.testing.assert_close(a,c,rtol=1e-3,atol=1e-5)
  for a,c in zip(*gradients):torch.testing.assert_close(a,c,rtol=2e-3,atol=2e-5)
