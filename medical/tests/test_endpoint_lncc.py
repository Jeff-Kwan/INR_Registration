"""Synthetic explicit-window values, gradients and physical-coordinate sampling."""
import unittest
from types import SimpleNamespace
import torch
from medical.registration.endpoint_lncc import window_loss, sample_window_batch
from medical.registration.ncc import correlation
from medical.registration.objective import endpoint_loss
from medical.models.siren import PairedSiren


class TestEndpointLNCC(unittest.TestCase):
    def test_window_reference_and_gradients(self):
        torch.manual_seed(3)
        a = torch.randn(4 * 27, dtype=torch.float64, requires_grad=True)
        b = torch.randn_like(a)
        actual = window_loss(a, b, 3, 1e-8)
        expected = torch.stack([1-correlation(x,y,1e-8) for x,y in zip(a.reshape(4,27),b.reshape(4,27))]).mean()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual,a)[0],torch.autograd.grad(expected,a)[0])
        self.assertTrue(torch.isfinite(window_loss(a*0,b*0,3,1e-8)))

    def test_sampling_physical_geometry_and_endpoint_gradients(self):
        torch.manual_seed(4)
        image = torch.arange(17.).view(1,1,17).expand(17,17,17).contiguous()
        branch = SimpleNamespace(origin=torch.ones(3)*2, domain_extent=torch.ones(3)*12, extent=torch.ones(3)*16)
        batch = sample_window_batch(branch,image,image,torch.Generator().manual_seed(4),
                                    {'windows':2,'size':3,'spacing_mm':1.},.5)
        points=(batch.coordinates+1)*8
        self.assertTrue(bool(((points>=2)&(points<=14)).all()))
        torch.testing.assert_close(batch.source,points[:,0])
        patches=points.reshape(2,3,3,3,3)
        torch.testing.assert_close(patches[:,1,0,0,0]-patches[:,0,0,0,0],torch.ones(2)*2)
        pair=PairedSiren(3,width=8,layers=2,omega=3.)
        options={'endpoint':'lncc','endpoint_lncc':{'size':3},'ncc':{'epsilon':1e-8,'moment_weight':.01}}
        terms=endpoint_loss(pair,batch,options)
        from medical.registration.ncc import endpoint_statistics
        a,b=pair.source(batch.coordinates).flatten(),pair.target(batch.coordinates).flatten()
        stats=endpoint_statistics(a,b,batch.source,batch.target,epsilon=1e-8,moment_weight=.01)
        expected=.5*(stats['endpoint_ncc_loss']+window_loss(a,batch.source,3,1e-8)
                     +window_loss(b,batch.target,3,1e-8))+stats['weighted_moments']
        torch.testing.assert_close(terms['total'],expected)
        parameters=tuple(pair.parameters())
        gradients=torch.autograd.grad(terms['total'],parameters)
        reference=torch.autograd.grad(expected,parameters)
        for actual, wanted in zip(gradients,reference):
            torch.testing.assert_close(actual,wanted)

        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
