"""Inverse-metric formula and live-gradient checks; no optimization steps."""
import copy
import os
import unittest
from unittest.mock import patch
import torch
from medical.configuration import load_config,validate_config
from medical.registration.symmetric_registration import shared_pair_velocity,shared_pair_inverse_metric,_pair_velocity_chunk
from medical.tests.test_siren_derivatives import make_pair
from medical.tests import test_siren_derivatives as fixtures

class TestInverseMetric(unittest.TestCase):
    def probes(self):
        torch.manual_seed(42)
        return tuple(torch.randn(*s,dtype=torch.float64,requires_grad=True) for s in
                     ((2,4,3),(2,4),(2,4,3,3),(2,4,3)))

    def test_literal_formula_gradient_and_velocity_agreement(self):
        g,q,h,m=self.probes()
        v=shared_pair_velocity(g,q,h,m,4.,1e-6)
        actual=shared_pair_inverse_metric(g,q,h,m,v,4.,1e-6)
        Q=g[..., :,None]*g[...,None,:]+4*h.transpose(-1,-2)@h
        b=g*q[...,None]+4*(h.transpose(-1,-2)@m[...,None]).squeeze(-1)
        A=Q.mean(1)+1e-6*torch.eye(3,dtype=g.dtype)
        force=(Q@v[:,None,:,None]).squeeze(-1)+b
        z=force-force.mean(1,keepdim=True)
        expected=torch.stack([sum(z[c,j]@torch.linalg.solve(A[c],z[c,j]) for j in range(4))/4 for c in range(2)])
        torch.testing.assert_close(actual,expected)
        left=torch.autograd.grad(actual.sum(),(g,q,h,m),retain_graph=True)
        right=torch.autograd.grad(expected.sum(),(g,q,h,m))
        for a,b in zip(left,right):torch.testing.assert_close(a,b)
        def loss(*inputs):
            v=shared_pair_velocity(*inputs,4.,1e-6)
            return shared_pair_inverse_metric(*inputs,v,4.,1e-6)
        self.assertTrue(torch.autograd.gradcheck(loss,(g,q,h,m),atol=1e-5,rtol=1e-4))

    def test_identical_probes_zero_and_permutation_invariant(self):
        p=self.probes();v=shared_pair_velocity(*p,4.,1e-6)
        a=shared_pair_inverse_metric(*p,v,4.,1e-6)
        perm=[3,1,0,2];changed=tuple(x[:,perm] for x in p)
        torch.testing.assert_close(a,shared_pair_inverse_metric(*changed,v,4.,1e-6))
        same=tuple(x[:,:1].expand_as(x) for x in p)
        z=shared_pair_inverse_metric(*same,shared_pair_velocity(*same,4.,1e-6),4.,1e-6)
        torch.testing.assert_close(z,torch.zeros_like(z),atol=1e-20,rtol=0)

    def test_registration_fields_unchanged_and_both_networks_live(self):
        pair=make_pair(width=4,omega=10.)
        branch=fixtures.TestLiveRegistration().branch();branch.resample_queries()
        before,_=branch._field_terms(pair,.37)
        branch.regularizer='inverse_metric'
        after,energy=branch._field_terms(pair,.37,return_energy=True)
        torch.testing.assert_close(before,after)
        with patch('medical.registration.symmetric_registration.shared_pair_energy',side_effect=AssertionError('PD-LK penalty must not run')):
            terms=branch.loss(pair)
        self.assertIn('inverse_metric',terms)
        self.assertNotIn('grid_pdlk',terms)
        gradients=torch.autograd.grad(energy,tuple(pair.parameters()))
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients),0)

    def test_config(self):
        self.assertEqual(load_config()['objective']['weight'],.1)
        config=load_config('medical/configs/inverse_metric.json')
        self.assertEqual(config['objective']['weight'],1.)
        lower=copy.deepcopy(config);lower['objective']['registration']['weight']=.01
        validate_config(lower)
        bad=copy.deepcopy(config);bad['objective']['registration']['registration_similarity']='gw_ncc'
        with self.assertRaises(ValueError):validate_config(bad)

    @unittest.skipUnless(os.environ.get('INR_TEST_CUDA')=='1','Opt-in compiled CUDA validation')
    def test_compiled_full_gradient_parity(self):
        pair=make_pair(width=8,omega=10.,dtype=torch.float32).cuda()
        coords=torch.rand(4,2,2,3,device='cuda')
        scale=coords.new_tensor([.13,.27,.51]);t=coords.new_tensor(.37)
        def call(function):
            p=pair.prepare().at(t)
            return function(p.parameters,p.direction,p.omegas,coords,scale,4.,1e-6,True,regularizer='inverse_metric')
        eager=call(_pair_velocity_chunk);compiled=call(torch.compile(_pair_velocity_chunk,fullgraph=True))
        for a,b in zip(eager,compiled):torch.testing.assert_close(a,b,rtol=1e-3,atol=1e-5)
        a=torch.autograd.grad(sum(x.square().sum() for x in eager),tuple(pair.parameters()))
        b=torch.autograd.grad(sum(x.square().sum() for x in compiled),tuple(pair.parameters()))
        for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=1e-3,atol=1e-4)
