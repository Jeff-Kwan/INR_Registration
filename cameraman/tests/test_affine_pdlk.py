"""Check elimination of the local velocity Jacobian, including its backward path."""
import unittest
import torch
from cameraman.core.affine_pdlk import adaptive_transport_energy,affine_temporal_loss
from cameraman.core.inr import Siren, pair_objective


def joint_energy(g,q,h,m,eta,c,beta):
    n,d=g.shape
    eye=torch.eye(d,dtype=g.dtype)
    k=torch.einsum('ni,jk->njik',g,eye).reshape(n,d,d*d)
    j=torch.cat((h,k),-1)
    design=torch.cat((g,torch.zeros(n,d*d,dtype=g.dtype)),-1)
    penalty=torch.tensor([eta]*d+[beta]*(d*d),dtype=g.dtype)
    matrix=design[:,:,None]*design[:,None,:]+c*j.transpose(-1,-2)@j+torch.diag(penalty)
    rhs=-(design*q[:,None]+c*(j.transpose(-1,-2)@m[:,:,None]).squeeze(-1))
    z=torch.linalg.solve(matrix,rhs[:,:,None]).squeeze(-1)
    residual=m+(j@z[:,:,None]).squeeze(-1)
    energy=((q+(design*z).sum(-1)).square()+c*residual.square().sum(-1)+(penalty*z.square()).sum(-1)).mean()
    return energy,z[:,:d]


class AffinePDLKTests(unittest.TestCase):
    def test_joint_elimination_and_gradients_2d_3d(self):
        for d in (2,3):
            for averaged in (False,True):
                torch.manual_seed(7)
                values=[torch.randn(*s,dtype=torch.float64,requires_grad=True) for s in ((7,d),(7,),(7,d,d),(7,d))]
                eta,c,beta=.03,4.,.2
                loss,a,weight=adaptive_transport_energy(*values,eta,c,beta,averaged)
                reference,velocity=joint_energy(*values,eta,c,beta+eta*c if averaged else beta)
                torch.testing.assert_close(loss,reference,rtol=1e-10,atol=1e-10)
                torch.testing.assert_close(a,velocity,rtol=1e-10,atol=1e-10)
                actual=torch.autograd.grad(loss,values,retain_graph=True)
                expected=torch.autograd.grad(reference,values)
                for x,y in zip(actual,expected):torch.testing.assert_close(x,y,rtol=1e-8,atol=1e-10)
                self.assertTrue(((weight>0)&(weight<=c)).all())

    def test_detached_and_differentiated_solve(self):
        torch.manual_seed(8)
        values=[torch.randn(*s,dtype=torch.float64,requires_grad=True) for s in ((5,2),(5,),(5,2,2),(5,2))]
        gradients=[]
        for differentiate in (False,True):
            loss,_,_=adaptive_transport_energy(*values,.01,4.,.1,differentiate_solve=differentiate)
            gradients.append(torch.autograd.grad(loss,values,retain_graph=True))
        for a,b in zip(*gradients):torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-10)

    def test_flat_regions_and_zero_c(self):
        g=torch.zeros(4,3,dtype=torch.float64)
        q=torch.ones(4,dtype=torch.float64)
        h=torch.zeros(4,3,3,dtype=torch.float64)
        m=torch.ones_like(g)
        loss,a,weight=adaptive_transport_energy(g,q,h,m,1e-6,4.,1e-3)
        torch.testing.assert_close(loss,torch.tensor(13.,dtype=loss.dtype))
        torch.testing.assert_close(weight,torch.full_like(q,4.))
        self.assertTrue(torch.isfinite(a).all())
        loss,_,weight=adaptive_transport_energy(g,q,h,m,1e-6,0.,1e-3)
        torch.testing.assert_close(loss,torch.tensor(1.,dtype=loss.dtype))
        self.assertEqual(weight.sum().item(),0.)

    def test_endpoint_gradients_and_finite_difference(self):
        torch.manual_seed(13)
        source,target=Siren(8,2).double(),Siren(8,2).double()
        coordinates=torch.rand(6,1,2,dtype=torch.float64)*2-1
        def evaluate():return affine_temporal_loss(source,target,coordinates,.37,(16,16),ridge=.01,beta=.05)
        loss=evaluate()
        loss.backward()
        for model in (source,target):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertGreater(sum(p.grad.square().sum().item() for p in model.parameters()),0.)
        parameter=source.net[0].linear.weight
        analytic=parameter.grad[0,0].item()
        with torch.no_grad():
            original=parameter[0,0].item()
            parameter[0,0]=original+1e-6
        plus=evaluate().item()
        with torch.no_grad():parameter[0,0]=original-1e-6
        minus=evaluate().item()
        with torch.no_grad():parameter[0,0]=original
        self.assertAlmostEqual(analytic,(plus-minus)/2e-6,delta=max(1e-7,abs(analytic)*1e-5))

    def test_paired_objective_endpoint_and_relaxation_gradients(self):
        torch.manual_seed(81)
        a, b = Siren(8, 2).double(), Siren(8, 2).double()
        xy = torch.rand(9, 2, dtype=torch.float64)
        first, second = torch.rand(2, 9, 1, dtype=torch.float64)
        points = torch.rand(5, 1, 2, dtype=torch.float64)
        for rho in (0., .4):
            def objective():
                return pair_objective(a, b, xy, first, second, points, .37,
                                      (16, 16), lk_ridge=.01, lk_affine_rho=rho)['total_loss']
            loss = objective()
            parameters = tuple(a.parameters()) + tuple(b.parameters())
            gradients = torch.autograd.grad(loss, parameters)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            parameter = a.net[0].linear.weight
            original = parameter[0, 0].item()
            with torch.no_grad():
                parameter[0, 0] = original + 1e-6
            plus = objective().item()
            with torch.no_grad():
                parameter[0, 0] = original - 1e-6
            minus = objective().item()
            with torch.no_grad():
                parameter[0, 0] = original
            self.assertAlmostEqual(gradients[0][0, 0].item(), (plus-minus)/2e-6, delta=1e-6)
        baseline = pair_objective(a, b, xy, first, second, None, None, (16, 16), lk_weight=0)
        expected = ((a(xy)-first).square() + (b(xy)-second).square()).mean()
        torch.testing.assert_close(baseline['total_loss'], expected)
        self.assertEqual(baseline['lk_loss'].item(), 0.)

    def test_invalid_beta(self):
        for beta in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                adaptive_transport_energy(torch.ones(2,2),torch.ones(2),torch.ones(2,2,2),torch.ones(2,2),1e-6,4.,beta)


if __name__=='__main__':unittest.main()
