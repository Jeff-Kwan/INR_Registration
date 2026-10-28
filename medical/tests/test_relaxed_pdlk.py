"""Cross-implementation rho equivalence and parameter-gradient checks."""
import unittest
from unittest.mock import patch
import torch
import json

from cameraman.core.affine_pdlk import adaptive_transport_energy
from medical.registration import pdlk
from medical.registration.train import validate_config, DEFAULT_CONFIG


class RelaxedTests(unittest.TestCase):
    def test_rho_beta_and_3d_energy_gradients(self):
        torch.manual_seed(7)
        for rho in (0., .4, 40.):
            for detach in (False, True):
                g=torch.randn(5,1,3,dtype=torch.float64,requires_grad=True)
                q=torch.randn(5,1,dtype=torch.float64,requires_grad=True)
                h=torch.randn(5,1,3,3,dtype=torch.float64,requires_grad=True)
                m=torch.randn(5,1,3,dtype=torch.float64,requires_grad=True)
                values=(g,q,h,m)
                reference=adaptive_transport_energy(*values,.01,4.,rho=rho,differentiate_solve=detach)[0]
                with patch.object(pdlk,'analytical_siren_derivatives',return_value=values):
                    result=pdlk.lk_temporal_loss(None,None,torch.zeros(5,1,3),.3,(8,9,10),
                        ridge=.01,gradient_weight=4.,affine_rho=rho,differentiate_solve=detach)
                torch.testing.assert_close(result,reference)
                first=torch.autograd.grad(result,values,retain_graph=True)
                second=torch.autograd.grad(reference,values,retain_graph=True)
                for a,b in zip(first,second):
                    torch.testing.assert_close(a,b)
                    self.assertTrue(torch.isfinite(a).all())
                if rho:
                    beta=adaptive_transport_energy(*values,.01,4.,4./rho,differentiate_solve=detach)[0]
                    torch.testing.assert_close(result,beta)
                else:
                    standard=pdlk.minimized_differential_transport_energy(*values,.01,4.,differentiate_solve=detach)[0]
                    torch.testing.assert_close(result,standard)

    def test_rho_config_validation(self):
        for rho in (-1., float('inf'), float('nan')):
            config=json.loads(DEFAULT_CONFIG.read_text())
            config['lk']['affine_rho']=rho
            with self.assertRaises(ValueError): validate_config(config)
        config=json.loads(DEFAULT_CONFIG.read_text())
        config['lk'].update(affine_rho=.4,affine_beta=10.)
        with self.assertRaises(ValueError): validate_config(config)


if __name__=='__main__': unittest.main()
