"""Synthetic endpoint GW-NCC composition/gradient checks; no optimizer steps."""
import unittest
import torch
from medical.models.siren import PairedSiren
from medical.registration.objective import TrainingBatch,endpoint_loss
from medical.registration.ncc import endpoint_statistics
from medical.registration.symmetric_registration import gaussian_weighted_ncc,source_grid_indices

class TestEndpointGWNCC(unittest.TestCase):
    def test_value_and_gradients_match_existing_window_loss(self):
        for size in (4,8):
            with self.subTest(grid_size=size):
                self.check_grid(size)

    def check_grid(self,size):
        torch.manual_seed(42)
        pair=PairedSiren(3,width=8,layers=2,omega=3.).double()
        coordinates=2*torch.rand(256,3,dtype=torch.float64)-1
        truth=torch.sin(coordinates[:,0])+coordinates[:,1].square()
        cells=source_grid_indices(coordinates,-torch.ones(3),2*torch.ones(3),1.,grid_size=size)
        batch=TrainingBatch(coordinates,truth,truth*.7+.2,source_cells=cells,target_cells=cells)
        settings={'endpoint':'gw_ncc','endpoint_gw_grid_size':size,'ncc':{'moment_weight':.01,'epsilon':1e-8}}
        actual=endpoint_loss(pair,batch,settings)
        a,b=pair.source(coordinates).flatten(),pair.target(coordinates).flatten()
        moments=endpoint_statistics(a,b,batch.source,batch.target,epsilon=1e-8,moment_weight=.01)
        expected=gaussian_weighted_ncc(a,batch.source,cells,grid_size=size,gaussian_kernel='three_tap')[0]
        expected+=gaussian_weighted_ncc(b,batch.target,cells,grid_size=size,gaussian_kernel='three_tap')[0]
        expected+=moments['weighted_moments']
        torch.testing.assert_close(actual['total'],expected)
        parameters=tuple(pair.parameters())
        gradients=torch.autograd.grad(actual['total'],parameters)
        reference=torch.autograd.grad(expected,parameters)
        for a,b in zip(gradients,reference):
            torch.testing.assert_close(a,b)
            self.assertTrue(torch.isfinite(a).all())

    def test_missing_cells_rejected(self):
        pair=PairedSiren(3,width=4,layers=1)
        points=torch.zeros(8,3)
        with self.assertRaises(ValueError):
            endpoint_loss(pair,TrainingBatch(points,torch.zeros(8),torch.zeros(8)),{'endpoint':'gw_ncc'})
