"""Protocol/map/launcher checks only: fitting and recovery are always mocked."""
import io
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from medical import brain_data, brain_evaluation as evaluation, brain_method
from medical.brain_experiments import main, make_jobs
from medical.configuration import load_config


class TestBrainProtocols(unittest.TestCase):
    def test_frozen_pairs(self):
        x = brain_data.pair_list('ixi')
        p = brain_data.pair_list('lpba40')
        self.assertEqual(len(x), 115)
        self.assertEqual(len(brain_data.pair_list('ixi', 'validation')), 58)
        self.assertEqual(len(p), 45)
        self.assertEqual(len(brain_data.pair_list('lpba40', 'validation')), 10)
        self.assertTrue(all(row['moving'] == 'atlas' for row in x))
        self.assertEqual(p[0], {'moving': 'S11', 'fixed': 'S12'})
        self.assertEqual(p[-1], {'moving': 'S34', 'fixed': 'S38'})
        self.assertEqual(len({frozenset(row.values()) for row in p}), 45)
        for dataset in ('ixi', 'lpba40'):
            test = {n for row in brain_data.pair_list(dataset) for n in row.values()} - {'atlas'}
            val = {n for row in brain_data.pair_list(dataset, 'validation') for n in row.values()} - {'atlas'}
            self.assertFalse(test & val)

    def test_pickle_reader_rejects_unrelated_classes(self):
        with self.assertRaises(pickle.UnpicklingError):
            brain_data.ArraysOnlyUnpickler(io.BytesIO(pickle.dumps(Path('/tmp')))).load()

    def test_default_plan_cannot_start_a_worker(self):
        with tempfile.TemporaryDirectory() as tmp, patch('medical.brain_experiments.GPUWorker') as worker:
            output = Path(tmp) / 'not_created'
            with patch('sys.stdout', new_callable=io.StringIO):
                plan = main(['--data-root', str(Path(tmp) / 'data'), '--output', str(output), '--plan-only'])
            self.assertFalse(plan['execution'])
            self.assertEqual(plan['counts'], {'ixi': 115, 'lpba40': 45})
            self.assertEqual(plan['single_seed'], 42)
            self.assertEqual(plan['config']['model']['width'], 512)
            self.assertEqual(plan['config']['training']['integrator'], 'rk2')
            self.assertEqual(plan['config']['training']['lr'], 1e-4)
            self.assertEqual(plan['config']['training']['optimizer']['name'], 'adamw')
            self.assertFalse(output.exists())
            worker.assert_not_called()

    def test_pairs_need_one_dataset(self):
        config = Path(brain_data.__file__).parent / 'configs/default.json'
        with self.assertRaises(ValueError):
            make_jobs(['ixi', 'lpba40'], 'test', Path('/tmp'), Path('/tmp/out'), config, [1])


class TestBrainMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_ixi_streamed_warp_matches_full_author_operation(self):
        shape = (7, 8, 9)
        ids = np.asarray(list(map(int, brain_data.protocols()['ixi_transmorph']['freesurfer_id_to_consecutive_id'])))
        labels = ids[(np.indices(shape).sum(axis=0) * 7) % len(ids)]
        mapping = evaluation.identity(shape, 'cpu')
        mapping[..., 0] += .49
        mapping[..., 1] -= .73
        mapping[-1, ..., 2] += 15  # all-channel zero ties select background
        source = torch.as_tensor(evaluation.remap_ixi(labels).astype(np.int64))
        one_hot = F.one_hot(source, num_classes=46).permute(3, 0, 1, 2)[None].float()
        normalized = 2 * mapping / mapping.new_tensor(np.asarray(shape) - 1) - 1
        reference = F.grid_sample(one_hot, normalized[..., [2, 1, 0]][None],
                                  align_corners=True, mode='bilinear').argmax(1)[0].numpy()
        self.assertTrue(np.array_equal(evaluation.warp_ixi(labels, mapping, class_chunk=7), reference))
        self.assertTrue(np.array_equal(evaluation.warp_ixi(labels, mapping, class_chunk=1), reference))

    def test_ixi_group_weighting_and_empty_labels(self):
        shape = (6, 7, 8)
        labels = np.full(shape, 2, dtype=np.uint8)  # left cerebral white matter only
        score = evaluation.ixi_direction(labels, labels, evaluation.identity(shape, 'cpu'))
        expected_dice = 2 * labels.size / (2 * labels.size + 1e-5)
        self.assertAlmostEqual(score['DSC']['mean'], expected_dice / 2 / 17)
        self.assertEqual(score['folding_fraction'], 0)
        self.assertEqual(score['DSC']['groups']['CSF'], 0)

    def test_lpba_nearest_and_invalid_samples(self):
        from medical.protocols.sitreg_reference import SITRegMetrics
        from composable_mapping import CoordinateSystem, DataFormat, LinearInterpolator, samplable_volume
        shape = (7, 8, 9)
        grid = evaluation.identity(shape, 'cpu')
        labels = (np.indices(shape)[0] + 21).astype(np.float32)
        mapping = grid.clone()
        mapping[..., 0] += 1.2
        ddf = (mapping - grid).movedim(-1, 0)
        reference = SITRegMetrics({'one': [21]})
        warped = reference._transform_mask(torch.from_numpy(labels)[None], ddf)[0].numpy()
        expected = np.zeros(shape, dtype=np.float32)
        expected[:-1] = labels[1:]
        self.assertTrue(np.array_equal(warped, expected))

    def test_lpba_sampled_jacobian_and_absent_label_policy(self):
        shape = (7, 8, 9)
        grid = evaluation.identity(shape, 'cpu')
        labels = ((np.indices(shape).sum(axis=0) % 2) + 21).astype(np.uint8)
        scores = evaluation.lpba_direction(labels, labels, grid, pair_index=2, jacobian_samples=512)
        self.assertEqual(scores['DSC']['mean'], 1)
        self.assertEqual(scores['HD95']['mean'], 0)
        self.assertEqual(scores['folding_fraction'], 0)
        self.assertIn('brainstem', scores['DSC']['excluded_nonfinite_labels'])
        reflected = grid.clone()
        reflected[..., 0] = shape[0] - 1 - reflected[..., 0]
        scores = evaluation.lpba_direction(labels, labels, reflected, pair_index=2, jacobian_samples=512)
        self.assertEqual(scores['folding_fraction'], 1)
        self.assertLess(scores['determinant_std'], 1e-6)
        self.assertEqual(scores['jacobian_seed'], 8524561973 ^ 1)
        json.dumps(scores, allow_nan=False)


class TestBrainMethod(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for directory in ('images', 'labels'):
            (self.root / directory).mkdir()
        self.shape = (7, 8, 9)
        self.config = load_config()
        self.pair = brain_data.pair_list('ixi')[0]
        meta = {'format_version': 1, 'dataset': 'ixi', 'shape': list(self.shape),
                'affine': np.eye(4).tolist(), 'intensity': 'released unchanged',
                'preparation_protocol_sha256': brain_data.preparation_fingerprint('ixi')}
        (self.root / 'protocol.json').write_text(json.dumps(meta))
        self.case = {}
        for index, key in enumerate(('moving', 'fixed')):
            data = np.arange(np.prod(self.shape), dtype=np.float32).reshape(self.shape) / 30 + index
            name = self.pair[key] + '.nii'
            path = self.root / 'images' / name
            nib.save(nib.Nifti1Image(data, np.eye(4)), path)
            nib.save(nib.Nifti1Image(np.full(self.shape, 2, dtype=np.uint8), np.eye(4)), self.root / 'labels' / name)
            self.case[key] = {'data': data, 'affine': np.eye(4), 'path': str(path)}
        self.shape_patch = patch.dict(brain_data.EXPECTED_SHAPES, {'ixi': self.shape})
        self.shape_patch.start()
        self.addCleanup(self.shape_patch.stop)

    def test_preserves_intensities_directions_and_cleanup(self):
        self.config['runtime']['recovery_max_axis'] = 256
        dummy = SimpleNamespace(model_config=self.config['model'], state_dict=lambda: {'w': torch.tensor(0.)})
        grid = evaluation.identity(tuple(reversed(self.shape)), 'cpu').flip(-1)
        scratch_paths = []
        def recover(pair, shape, settings, scratch, **kwargs):
            self.assertEqual(kwargs['max_axis'], 256)
            scratch_paths.append(Path(scratch))
            (Path(scratch) / 'temporary.txt').write_text('scratch')
            return SimpleNamespace(inverse=grid + torch.tensor([1., 2., 3.]), forward=grid,
                                   diagnostics=[{'converged': True}])
        with patch.object(brain_method, 'fit_pair', return_value=dummy) as fit, \
             patch.object(brain_method, 'recover', side_effect=recover), \
             patch.object(brain_method, 'evaluate', return_value={'DSC': {'mean': .5}}) as score:
            result = brain_method.run_case(self.case, self.config, 'cpu')
        self.assertTrue(torch.equal(fit.call_args.args[0], torch.from_numpy(self.case['moving']['data'].transpose(2, 1, 0).copy())))
        self.assertNotIn('moving_mask', fit.call_args.kwargs)
        self.assertNotIn('fixed_mask', fit.call_args.kwargs)
        expected = evaluation.identity(self.shape, 'cpu') + torch.tensor([1., 2., 3.])
        self.assertTrue(torch.equal(score.call_args.args[1], expected))
        self.assertFalse(any(p.exists() for p in scratch_paths))
        self.assertEqual(set(result['checkpoint']['state']), {'w'})
        json.dumps(result['metrics'], allow_nan=False)

    def test_cleanup_on_evaluation_exception(self):
        dummy = SimpleNamespace(model_config=self.config['model'])
        grid = evaluation.identity(tuple(reversed(self.shape)), 'cpu').flip(-1)
        paths = []
        def recover(*args, **kwargs):
            paths.append(Path(args[3]))
            return SimpleNamespace(inverse=grid, forward=grid, diagnostics=[])
        with patch.object(brain_method, 'fit_pair', return_value=dummy), \
             patch.object(brain_method, 'recover', side_effect=recover), \
             patch.object(brain_method, 'evaluate', side_effect=RuntimeError('evaluation failed')):
            with self.assertRaisesRegex(RuntimeError, 'evaluation failed'):
                brain_method.run_case(self.case, self.config, 'cpu')
        self.assertFalse(any(p.exists() for p in paths))


if __name__ == '__main__':
    unittest.main()
