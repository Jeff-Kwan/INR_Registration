"""Synthetic checks for optional structure sampling; no optimizer steps."""
import unittest
import torch
from medical.registration.symmetric_registration import SymmetricPairRegistration,sample_crop_indices

class TestStructureSampling(unittest.TestCase):
    def test_constant_falls_back_to_identical_uniform_draws(self):
        from medical.registration.endpoint_sampling import StructureEndpointSampler
        image=torch.ones(7,8,9)
        sampler=StructureEndpointSampler(image,None,torch.eye(4),{'structure_fraction':.5,'sigma_mm':2.})
        self.assertTrue(sampler.metadata['fallback_uniform'])
        actual=sampler.sample(100,torch.Generator().manual_seed(42))
        expected=torch.randint(image.numel(),(100,),generator=torch.Generator().manual_seed(42))
        torch.testing.assert_close(actual,expected)

    def test_mixture_distribution_and_independent_mask_support(self):
        import numpy as np
        from medical.registration.endpoint_sampling import StructureEndpointSampler,structure_scores
        image=torch.zeros(9,9,25);image[:,:,12:]=1
        options={'structure_fraction':.5,'sigma_mm':2.}
        sampler=StructureEndpointSampler(image,None,torch.eye(4),options)
        score=structure_scores(image,torch.eye(4),2.).reshape(-1)
        expected=.5/image.numel()+.5*score/score.sum()
        indices=sampler.sample(100000,torch.Generator().manual_seed(42))
        central=np.indices(image.shape)[2].reshape(-1)
        central=(central>=10)&(central<=14)
        actual=torch.from_numpy(central)[indices].float().mean().item()
        self.assertAlmostEqual(actual,float(expected[central].sum()),delta=.008)
        active=torch.arange(50)
        masked=StructureEndpointSampler(image,active,torch.eye(4),options)
        self.assertTrue(torch.isin(masked.sample(1000,torch.Generator().manual_seed(1)),active).all())
        torch.testing.assert_close(indices,sampler.sample(100000,torch.Generator().manual_seed(42)))

    def test_physical_derivatives_axis_order_rotation_and_reflection(self):
        import numpy as np
        from medical.registration.endpoint_sampling import structure_scores
        # Physical ramp dI/dx=1, sampled every 2 mm along tensor's last axis.
        image=(torch.arange(31)*2.).expand(11,11,31).contiguous()
        affine=np.diag([1.,1.,2.,1.])
        score=structure_scores(image,affine,2.)
        np.testing.assert_allclose(score[4:7,4:7,8:-8],1.,atol=.001)
        rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,-1.]])
        affine[:3,:3]=rotation@affine[:3,:3]
        np.testing.assert_allclose(structure_scores(image,affine,2.),score,atol=1e-6)

    def test_configuration_and_rejected_geometry(self):
        import copy
        from medical.configuration import DEFAULT,validate_config,load_config
        from medical.registration.endpoint_sampling import structure_scores
        load_config('medical/configs/physical_working_grid_structure.json')
        for options in ({'sigma_mm':2.}, {'sigma_mm':0.,'structure_fraction':.5},
                        {'sigma_mm':2.,'structure_fraction':float('nan')}):
            config=copy.deepcopy(DEFAULT);config['training']['endpoint_sampling']=options
            with self.assertRaises(ValueError):validate_config(config)
        affine=torch.eye(4);affine[0,1]=.2
        with self.assertRaises(ValueError):structure_scores(torch.zeros(5,5,5),affine,2.)
