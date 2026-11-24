"""Exercise coordinate/direction and cleanup boundaries, without native training."""
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import nibabel as nib
import numpy as np
import torch

from baselines import ccidir, convexadam, fireants, nir
from baselines.common import validate_config
from baselines.sources import ROOT


def config(name):
    return json.loads((ROOT / 'configs' / f'{name}_neurite_oasis.json').read_text())


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        directory = Path(self.temp.name)
        self.shape = (5, 7, 9)  # Unequal dimensions catch XYZ/ZYX swaps.
        self.points = np.array([[1.2, 2.3, 3.4], [2., 4., 6.]])
        self.shift = np.array([0.2, -0.4, 0.6])
        affine = np.diag([-0.97, -1.13, 2.5, 1.0])
        self.case = {}
        for key in ('fixed', 'moving'):
            data = np.arange(np.prod(self.shape), dtype=np.float32).reshape(self.shape)
            path = directory / f'{key}.nii.gz'
            self.case[key] = {'data': data, 'affine': affine, 'path': str(path)}
        self.enterContext(patch('medical.comparison_metrics.validate_case'))
        self.enterContext(patch('medical.comparison_metrics.evaluate', side_effect=self.score))

    def score(self, case, predict, evaluation):
        self.assertEqual(evaluation, 'neurite_oasis')
        error = np.abs(predict(self.points) - (self.points + self.shift)).max()
        return {'max_coordinate_error': float(error)}

    def contexts(self, module, native):
        self.enterContext(patch.object(module, 'upstream_imports', return_value=nullcontext()))
        self.enterContext(patch.object(module, 'execution', return_value=nullcontext(torch.device('cpu'))))
        self.enterContext(patch.object(module, 'native_modules', return_value=native))

    def grid(self):
        axes = torch.meshgrid(*[torch.arange(n, dtype=torch.float32) for n in self.shape[::-1]], indexing='ij')
        xyz = torch.stack(axes[::-1], -1)
        return (2 * (xyz + torch.tensor(self.shift)) / torch.tensor(np.array(self.shape)-1) - 1).unsqueeze(0)

    def assert_scores(self, result):
        self.assertLess(result['metrics']['max_coordinate_error'], 2e-6)

    def test_convexadam_native_voxel_displacement(self):
        native = SimpleNamespace(convex_adam_pt=Mock(return_value=np.broadcast_to(self.shift, (*self.shape, 3))))
        self.contexts(convexadam, [native])
        result = convexadam.run_case(self.case, config('convexadam'), 'cpu')
        self.assert_scores(result)
        self.assertFalse(result['checkpoint']['parameter_checkpoint'])
        call = native.convex_adam_pt.call_args.kwargs
        self.assertIs(call['img_fixed'], self.case['fixed']['data'])
        self.assertTrue(call['ic'])
        self.assertNotIn('displacement', result['checkpoint'])

    def test_single_case_runner_persists_only_compact_evidence(self):
        from medical.case_runner import run_case
        native = SimpleNamespace(convex_adam_pt=Mock(return_value=np.broadcast_to(self.shift, (*self.shape, 3))))
        self.contexts(convexadam, [native])
        for volume in self.case.values():
            nib.save(nib.Nifti1Image(volume['data'], volume['affine']), volume['path'])
        output = Path(self.temp.name) / 'output'
        scores = run_case(self.case['moving']['path'], self.case['fixed']['path'],
                          config('convexadam'), output, convexadam.run_case, 'cpu')
        self.assertLess(scores['max_coordinate_error'], 2e-6)
        self.assertEqual({path.name for path in output.iterdir()}, {'checkpoint.pt', 'metrics.json', 'run.json'})
        checkpoint = torch.load(output / 'checkpoint.pt', weights_only=True)
        self.assertEqual(set(checkpoint), {'method', 'source', 'config', 'kind', 'parameter_checkpoint', 'reason'})

    def test_fireants_uses_native_syn_composition(self):
        reg = SimpleNamespace(optimize=Mock(), get_warped_coordinates=Mock(return_value=self.grid()))
        image = SimpleNamespace(Image=SimpleNamespace(load_file=Mock(side_effect=lambda path, **kw: path)), BatchedImages=lambda images: images)
        syn = SimpleNamespace(SyNRegistration=Mock(return_value=reg))
        self.contexts(fireants, [image, syn])
        self.enterContext(patch.dict('sys.modules', {'fireants.interpolator': SimpleNamespace(fireants_interpolator=SimpleNamespace(use_ffo=False))}))
        result = fireants.run_case(self.case, config('fireants'), 'cpu')
        self.assert_scores(result)
        reg.optimize.assert_called_once_with()  # This is a Mock; no optimizer runs.
        reg.get_warped_coordinates.assert_called_once()
        self.assertFalse(result['checkpoint']['parameter_checkpoint'])

    def cc_native(self, fail=False):
        network = torch.nn.Linear(3, 3)
        reverse = torch.nn.Linear(3, 3)
        self.cc_scratch = []
        def constructor(moving, fixed, **kwargs):
            self.cc_scratch.append(Path(kwargs['save_folder']))
            self.assertEqual(tuple(fixed.shape), self.shape[::-1])
            return SimpleNamespace(network=network, network_rev=reverse, fit=Mock(side_effect=RuntimeError('native failure') if fail else None))
        self.native_coordinates = []
        def consensus(net, rev, coordinates, inf_order):
            self.assertEqual(inf_order, 2)
            self.native_coordinates.append(coordinates.detach().numpy())
            delta = torch.tensor(4 * self.shift[::-1].copy() / (np.array(self.shape[::-1]) - 1), dtype=torch.float32)
            # A nonzero consensus offset catches using forward-only inference.
            return coordinates, coordinates, delta.expand_as(coordinates)
        general = SimpleNamespace(compute_coordinates_cycle_autoretry=Mock(side_effect=consensus))
        return [SimpleNamespace(ImplicitRegistrator=Mock(side_effect=constructor)), general]

    def test_ccidir_preserves_dense_cycle_consensus(self):
        self.contexts(ccidir, self.cc_native())
        self.enterContext(patch.object(torch.Tensor, 'cuda', lambda value: value))
        result = ccidir.run_case(self.case, config('ccidir'), 'cuda:0')
        self.assert_scores(result)
        expected = 2 * self.points[:, ::-1] / (np.array(self.shape[::-1]) - 1) - 1
        np.testing.assert_allclose(self.native_coordinates[0], expected, atol=1e-7)
        self.assertTrue(all(not path.exists() for path in self.cc_scratch))
        self.assertEqual(set(result['checkpoint']['forward']), {'weight', 'bias'})

    def test_ccidir_cleans_native_failure_scratch(self):
        self.contexts(ccidir, self.cc_native(fail=True))
        with self.assertRaisesRegex(RuntimeError, 'native failure'):
            ccidir.run_case(self.case, config('ccidir'), 'cuda:0')
        self.assertTrue(self.cc_scratch)
        self.assertTrue(all(not path.exists() for path in self.cc_scratch))

    def nir_native(self, fail=False):
        class Samples:
            def __init__(inner, dataset, **kwargs):
                inner.maxlen = dataset.shape
            def __len__(inner):
                return 8
        dataio = SimpleNamespace(Implicit3DWrapper=Samples, Single_CT_Brain_Model=Mock(),
                                normalize_coords=lambda points, shape, maxlen: torch.tensor(2*points/(np.array(shape)-1)-1, dtype=torch.float32),
                                unnormalize_coords=lambda points, shape, maxlen: (points.numpy()+1)*(np.array(shape)-1)/2)
        class Model(torch.nn.Module):
            def __init__(inner):
                super().__init__()
                inner.weight = torch.nn.Parameter(torch.zeros(1))
            def cuda(inner):
                return inner
            def deform(inner, coords):
                return coords * torch.tensor((np.array(self.shape)-1)/np.array(self.shape)) + torch.tensor(2*self.shift/np.array(self.shape))
        self.enterContext(patch.object(nir, 'build_model', return_value=Model()))
        writer = SimpleNamespace(close=Mock())
        self.nir_scratch = []
        def train(*args, **kwargs):
            self.nir_scratch.append(Path(kwargs['model_dir']).parent)
            training.SummaryWriter('mock-only')
            if fail:
                raise RuntimeError('native failure')
        training = SimpleNamespace(SummaryWriter=Mock(return_value=writer), train_efficient_pairwise=Mock(side_effect=train))
        self.nir_writer = writer
        return [dataio, None, None, None, training]

    def test_nir_native_loop_and_cleanup(self):
        native = self.nir_native()
        self.contexts(nir, native)
        self.enterContext(patch.object(torch.Tensor, 'cuda', lambda value: value))
        result = nir.run_case(self.case, config('nir'), 'cuda:0')
        self.assert_scores(result)
        kwargs = native[-1].train_efficient_pairwise.call_args.kwargs
        self.assertEqual(kwargs['epochs'], 900)
        self.assertEqual(kwargs['lr'], 1e-4)
        self.assertEqual(kwargs['loss_weight'], 1000)
        self.assertEqual(result['metrics']['requested_steps'], 900)
        self.assertEqual(result['metrics']['native_loop_steps'], 904)
        self.assertEqual(kwargs['loss_types'], ['ncc', 'discrete_jacobian'])
        self.assertEqual(set(result['checkpoint']['state']), {'weight'})
        self.assertTrue(all(not path.exists() for path in self.nir_scratch))
        self.nir_writer.close.assert_called_once()

    def test_nir_cleans_native_failure_scratch(self):
        self.contexts(nir, self.nir_native(fail=True))
        self.enterContext(patch.object(torch.Tensor, 'cuda', lambda value: value))
        with self.assertRaisesRegex(RuntimeError, 'native failure'):
            nir.run_case(self.case, config('nir'), 'cuda:0')
        self.assertTrue(all(not path.exists() for path in self.nir_scratch))
        self.nir_writer.close.assert_called_once()

    def test_configs_reject_ignored_typo(self):
        cfg = config('ccidir')
        cfg['native']['cycle_aplha'] = 0.001
        with self.assertRaises(ValueError):
            validate_config(cfg, 'ccidir')


if __name__ == '__main__':
    unittest.main()
