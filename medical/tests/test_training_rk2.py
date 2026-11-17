"""Analytic field and support checks only; no fitting or saved-model recovery."""
import unittest
import torch
from medical.registration.symmetric_registration import SymmetricPairRegistration,sample_crop_indices

class TestTrainingRK2(unittest.TestCase):
    def branch(self):
        image=torch.zeros(17,17,17)
        return SymmetricPairRegistration(image,image,torch.eye(4),grid_size=4,particles=32,
                                         compile_derivatives=False,training_integrator='rk2')

    def test_linear_field_midpoint_formula_and_full_gradient(self):
        branch=self.branch(); start=torch.tensor([[5.,6.,7.]])
        centers=(torch.arange(4)+.5)*4
        z,y,x=torch.meshgrid(centers,centers,centers,indexing='ij')
        coordinates=torch.stack((x,y,z),-1)
        a=torch.tensor(.1,requires_grad=True);b=torch.tensor(.2,requires_grad=True)
        fields=(a*coordinates,b*coordinates)
        for reverse in (False,True):
            result,valid=branch._trajectory(start,fields,reverse)
            h=-.5 if reverse else .5
            factor=(1+h*a+.5*(h*a)**2)*(1+h*b+.5*(h*b)**2)
            torch.testing.assert_close(result,start*factor)
            self.assertTrue(valid.all())
        grad=torch.autograd.grad(result.sum(),(a,b))
        self.assertTrue(all(torch.isfinite(g) and g.abs()>0 for g in grad))

    def test_constant_translation_and_support(self):
        branch=self.branch();field=torch.zeros(4,4,4,3);field[...,0]=2
        start=torch.tensor([[5.,6.,7.]])
        for reverse in (False,True):
            result,valid=branch._trajectory(start,(field,field),reverse)
            torch.testing.assert_close(result,start+torch.tensor([[-2. if reverse else 2.,0.,0.]]))
            self.assertTrue(valid.all())
        branch.cropped=True
        bad=field*100
        _,valid=branch._trajectory(start,(bad,bad),False)
        self.assertFalse(valid.any())

    def test_cropped_reconstruction_indices(self):
        branch=self.branch();branch.crop_shape=(9,9,9)
        indices=sample_crop_indices(branch,10000,torch.Generator().manual_seed(42))
        # Convert flattened indices independently into XYZ voxel coordinates.
        xyz=torch.stack((indices%17,(indices//17)%17,indices//(17*17)),-1).float()
        self.assertTrue(branch.supported(xyz).all())


    def test_diagnostics_do_not_change_registration_loss_or_gradients(self):
        from medical.models.siren import PairedSiren
        torch.manual_seed(42)
        image = torch.rand(11, 13, 15, dtype=torch.float64)
        pair = PairedSiren(width=8, layers=2, omega=10.).double()
        with torch.no_grad():
            for parameter in pair.target.parameters():
                parameter.add_(.001 * torch.randn_like(parameter))
        for similarity in ('gw_ncc',):
            results = []
            for diagnostics in (True, False):
                branch = SymmetricPairRegistration(
                    image, image.flip(0), torch.eye(4), grid_size=4,
                    particles=64, crop_fraction=.5, pairs_per_cell=2,
                    compile_derivatives=False, recompute=False,
                    gaussian_kernel='three_tap', registration_similarity=similarity)
                terms = branch.loss(pair, diagnostics=diagnostics)
                self.assertEqual('outside' in terms, diagnostics)
                loss = .1 * terms['grid_pdlk'] + .01 * terms['registration']
                gradients = torch.autograd.grad(loss, tuple(pair.parameters()))
                results.append((loss, *gradients))
            for expected, actual in zip(*results):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
