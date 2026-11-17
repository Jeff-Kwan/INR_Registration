"""Analytical L2 gradient check, with no fitting or recovery."""
import unittest
import torch
from medical.registration.objective import parameter_agreement
from medical.configuration import load_config, validate_config


class TestParameterAgreement(unittest.TestCase):
    def test_mean_over_weights_and_biases_and_exact_gradient(self):
        a=torch.tensor([[1.,2.,3.]],requires_grad=True)
        b=torch.tensor([4.],requires_grad=True)
        c=torch.tensor([[2.,4.,6.]],requires_grad=True)
        d=torch.tensor([8.],requires_grad=True)
        loss=parameter_agreement(((a,b),),((c,d),))
        torch.testing.assert_close(loss,torch.tensor(7.5))
        gradients=torch.autograd.grad(loss,(a,b,c,d))
        for got,expected in zip(gradients,((a-c)/2,(b-d)/2,(c-a)/2,(d-b)/2)):
            torch.testing.assert_close(got,expected)

    def test_configuration_guard(self):
        for value in (0.,1e-3,1.):
            config=load_config();config['objective']['parameter_l2']=value
            validate_config(config)
        for value in (-1.,float('inf'),True):
            config=load_config();config['objective']['parameter_l2']=value
            with self.assertRaises(ValueError):validate_config(config)
