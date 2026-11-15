"""Independent tiny-system PCG checks; no training or real-data recovery."""
import itertools
import unittest
from unittest.mock import patch
import torch
from medical.registration.flow import estimate_velocity_from_derivatives as solve, laplacian, stencil_laplacian, pcg_step

class TestFlow(unittest.TestCase):
    def matrix(self, g, smoothness, ridge):
        shape=g.shape[:-1];d=g.shape[-1];n=g.numel()
        a=torch.eye(n,dtype=g.dtype)*ridge
        points=list(itertools.product(*(range(s) for s in shape)))
        ids={p:i for i,p in enumerate(points)}
        for p,i in ids.items():
            a[i*d:(i+1)*d,i*d:(i+1)*d]+=torch.outer(g[p],g[p])
            for axis in range(d):
                q=list(p);q[axis]+=1;q=tuple(q)
                if q in ids:
                    for c in range(d):
                        x,y=i*d+c,ids[q]*d+c
                        a[x,x]+=smoothness;a[y,y]+=smoothness
                        a[x,y]-=smoothness;a[y,x]-=smoothness
        return a

    def test_direct_solutions_and_boundaries(self):
        torch.manual_seed(42)
        for shape in ((3,4),(2,3,4)):
            g=torch.randn(*shape,len(shape),dtype=torch.float64)
            q=torch.randn(*shape,dtype=torch.float64)
            a=self.matrix(g,.1,.05);b=(-g*q[...,None]).flatten()
            expected=torch.linalg.solve(a,b).reshape_as(g)
            for initial in (None,expected,torch.ones_like(g)):
                v,diag=solve(g,q,smoothness=.1,ridge=.05,tolerance=1e-10,initial_velocity=initial)
                torch.testing.assert_close(v,expected,rtol=1e-7,atol=1e-9)
                self.assertTrue(diag['converged'])
                self.assertLessEqual(float((b-a@v.flatten()).norm()/b.norm()),1e-10)
            for point in itertools.product(*(range(s) for s in shape)):
                v=torch.zeros_like(g);v[point]=1
                torch.testing.assert_close(stencil_laplacian(v),laplacian(v))

    def test_zero_rhs_and_iteration_limit(self):
        g=torch.ones(3,4,2)
        v,d=solve(g,torch.zeros(3,4),initial_velocity=g)
        self.assertEqual(v.count_nonzero(),0);self.assertEqual(d['iterations'],0)
        v,d=solve(g,torch.arange(12.).reshape(3,4),iterations=1,tolerance=1e-12)
        self.assertFalse(d['converged']);self.assertEqual(d['termination_reason'],'iteration_limit')

    def test_optimistic_recursive_residual_restarts(self):
        torch.manual_seed(42)
        g=torch.randn(4,5,2);q=torch.randn(4,5)
        def optimistic(*args):
            v,r,p,rz=pcg_step(*args)
            return v,torch.zeros_like(r),p,rz
        with patch('medical.registration.flow.pcg_step',side_effect=optimistic):
            _,d=solve(g,q,iterations=11,tolerance=1e-10)
        self.assertGreater(d['residual_restarts'],0)
        self.assertFalse(d['converged'])

    @unittest.skipUnless(__import__('os').environ.get('INR_TEST_CUDA')=='1','Opt-in CUDA validation')
    def test_compiled_fixed_iteration_parity(self):
        torch.manual_seed(42)
        g=torch.randn(9,7,5,3,device='cuda')*.1;q=torch.randn(9,7,5,device='cuda')
        a,da=solve(g,q,iterations=40,tolerance=1e-12,compile_solver=False)
        b,db=solve(g,q,iterations=40,tolerance=1e-12,compile_solver=True)
        self.assertEqual(da['iterations'],db['iterations'])
        self.assertLess(float((a-b).norm()/a.norm()),1e-3)

    @unittest.skipUnless(__import__('os').environ.get('INR_TEST_CUDA')=='1','Opt-in CUDA validation')
    def test_cuda_weighted_warm_start_and_noncontiguous_input(self):
        # Exercise a partial tile, singleton axis, and physical-spacing weights.
        torch.manual_seed(42)
        g = torch.randn(17, 1, 23, 3, device='cuda').transpose(0, 2)
        q = torch.randn(g.shape[:-1], device='cuda')
        start = torch.randn_like(g)
        saved = start.clone()
        options = dict(iterations=300, tolerance=2e-5, smoothness=.06, ridge=.05,
                       spacing=(.8, 1.3, 2.4), initial_velocity=start)
        expected, reference = solve(g, q, compile_solver=False, **options)
        actual, diagnostics = solve(g, q, compile_solver=True, **options)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(start, saved, rtol=0, atol=0)
        self.assertEqual(reference['converged'], diagnostics['converged'])
        self.assertTrue(diagnostics['converged'])
        self.assertLessEqual(diagnostics['relative_linear_residual'], options['tolerance'])
        zeros, diagnostics = solve(g, torch.zeros_like(q), compile_solver=True, **options)
        self.assertEqual(zeros.count_nonzero(), 0)
        self.assertEqual(diagnostics['iterations'], 0)

    def test_recovery_scratch_cleaned_on_error(self):
        import tempfile
        from pathlib import Path
        from medical.registration.recovery import recover
        from medical.configuration import load_config
        with tempfile.TemporaryDirectory() as scratch:
            with patch('medical.registration.recovery.derivative_grid',side_effect=RuntimeError('injected')):
                with self.assertRaisesRegex(RuntimeError,'injected'):
                    recover(torch.nn.Linear(1,1),(3,4,5),load_config()['recovery'],scratch)
            self.assertEqual(list(Path(scratch).iterdir()),[])
