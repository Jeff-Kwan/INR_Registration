"""Synthetic preparation/metric checks and mocked model/launcher execution."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch
from skimage.transform import rescale

from scripts.prepare_acdc import preprocess
from medical import cohort_method, cohort_experiments
from medical.cohort_evaluation import acdc_metrics, evaluate
from medical.configuration import load_config
from medical.registration.geometry import identity_grid
from medical.registration.recovery import RecoveryResult


class TestPreparedCohorts(unittest.TestCase):
    def test_acdc_preprocessing_matches_array_recipe(self):
        shape = (140, 142, 20)
        image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        labels = (np.indices(shape).sum(0) % 4).astype(np.float32)
        spacing = (1.5, 1.5, 3.15)
        actual, codes, offset = preprocess(image, labels, spacing)
        scaled = rescale(image, (1, 1, 1), order=3, mode='edge')
        scaled = (scaled - scaled.min()) / (scaled.max() - scaled.min())
        np.testing.assert_array_equal(offset, (6, 7, 0))
        np.testing.assert_array_equal(actual, scaled[6:134, 7:135])
        np.testing.assert_array_equal(codes, labels[6:134, 7:135])
        self.assertEqual(actual.shape, (128, 128, 20))
        self.assertGreater(float(actual.min()), 0)  # normalization precedes crop
        with self.assertRaises(ValueError):
            preprocess(np.ones(shape), labels, spacing)
        with self.assertRaises(ValueError):
            preprocess(image[:100], labels[:100], spacing)

    def test_acdc_preserves_float32_header_scale_arithmetic(self):
        shape = (150, 152, 12)
        image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        labels = (np.indices(shape).sum(0) % 4).astype(np.float32)
        spacing = np.array([1.36719, 1.36719, 8.0], dtype=np.float32)
        with patch('scripts.prepare_acdc.rescale', wraps=rescale) as resize:
            preprocess(image, labels, spacing)
        factors = resize.call_args_list[0].args[1]
        expected = tuple(spacing[i] / target for i, target in enumerate((1.5, 1.5, 3.15)))
        self.assertIsInstance(factors[0], np.float32)
        self.assertEqual(factors, expected)

    def test_author_missing_label_policy_and_hd95_spacing(self):
        truth = (np.indices((8, 9, 10))[0] // 2).astype(np.uint8)
        prediction = truth.copy()
        prediction[prediction == 2] = 0
        result = acdc_metrics(truth, prediction)
        self.assertEqual(result['labels'], [1, 3])
        self.assertEqual(result['DSC']['mean'], 1.)
        self.assertAlmostEqual(result['fixed_label_diagnostics']['DSC']['mean'], 2/3)
        self.assertIsNone(result['fixed_label_diagnostics']['HD95']['detailed'][1])
        shifted = np.roll(truth, 1, axis=0)
        self.assertEqual(acdc_metrics(truth, shifted)['HD95']['detailed'][0], 1.5)

    def test_acdc_xyz_mapping_zero_padding_and_folding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shape = (8, 9, 10)
            codes = (np.indices(shape)[0] // 2).astype(np.uint8)
            for name in ('fixed', 'moving'):
                nib.save(nib.Nifti1Image(codes, np.eye(4)), root / f'{name}.nii.gz')
            pair = {f'{k}_label': f'{k}.nii.gz' for k in ('moving', 'fixed')}
            grid = identity_grid(shape[::-1], 'cpu').permute(2, 1, 0, 3)
            scores = evaluate({}, grid, 'acdc', root, pair)
            self.assertEqual(scores['DSC']['mean'], 1)
            self.assertEqual(scores['folding_fraction'], 0)
            outside = grid + 100
            scores = evaluate({}, outside, 'acdc', root, pair)
            self.assertEqual(scores['scored_labels'], 0)
            self.assertIsNone(scores['DSC']['mean'])

    def test_adapter_preserves_intensities_and_cleans_scratch(self):
        for dataset, protocol in [('acdc', 'acdc_kan_idir_v1'), ('neurite_oasis', 'neurite_oasis_gaussiandir_last50_consecutive49_v1')]:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / 'manifest.json').write_text(json.dumps({'protocol': protocol}))
                values = np.arange(210, dtype=np.float32).reshape(5, 6, 7) / 300
                case = {k: dict(path=str(root / k / 'image.nii.gz'), data=values, affine=np.eye(4)) for k in ('moving', 'fixed')}
                pair = torch.nn.Linear(1, 1)
                pair.model_config = {'width': 1}
                grid = identity_grid((7, 6, 5), 'cpu')
                scratch_paths = []
                def recover(*args, **kwargs):
                    scratch = Path(args[3])
                    (scratch / 'working.bin').write_bytes(b'temporary')
                    scratch_paths.append(scratch)
                    return RecoveryResult(grid, grid, [])
                with patch.object(cohort_method, 'validate_case', return_value=(root, {'pair_index': 1})), \
                     patch.object(cohort_method, 'fit_pair', return_value=pair) as fit, \
                     patch.object(cohort_method, 'recover', side_effect=recover), \
                     patch.object(cohort_method, 'evaluate', return_value={}) as score:
                    result = cohort_method.run_case(case, load_config(), 'cpu')
                    np.testing.assert_array_equal(fit.call_args.args[0].numpy(), values.transpose(2, 1, 0))
                    torch.testing.assert_close(score.call_args.args[1], grid.permute(2, 1, 0, 3))
                    self.assertEqual(result['checkpoint']['cohort'], dataset)
                    score.side_effect = RuntimeError('evaluation failed')
                    with self.assertRaises(RuntimeError):
                        cohort_method.run_case(case, load_config(), 'cpu')
                self.assertTrue(all(not p.exists() for p in scratch_paths))

    def test_plan_is_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'uncreated'
            settings = load_config()
            with patch.object(cohort_experiments, 'make_jobs', return_value=([], settings)), \
                 patch.object(cohort_experiments, 'GPUWorker') as worker, redirect_stdout(io.StringIO()):
                plan = cohort_experiments.main(['--plan-only', '--output', str(output)])
            self.assertFalse(plan['execution'])
            self.assertFalse(output.exists())
            worker.assert_not_called()


class TestCohortQueue(unittest.TestCase):
    def test_failures_are_counted_and_remaining_jobs_finish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / 'config.json'
            settings = load_config()
            config.write_text(json.dumps(settings))
            output = root / 'output'
            jobs = [dict(dataset='acdc', pair_index=i, seed=42, output=str(output / f'pair{i}'),
                         missing_inputs=[], command=['python', '--moving', 'moving', '--fixed', 'fixed',
                         '--config', str(config), '--output', str(output / f'pair{i}')]) for i in range(1, 4)]
            visited = []
            def run(command, log):
                target = Path(command[command.index('--output') + 1])
                visited.append(target.name)
                Path(log).touch()
                if target.name == 'pair2':
                    return {'returncode': 1}
                target.mkdir()
                (target / 'metrics.json').write_text(json.dumps(dict(DSC=dict(mean=1., detailed=[1., 1., 1.]),
                    HD95=dict(mean=0.), folding_fraction=0., folding_percent=0., recovery=[dict(converged=True)])))
                return {'returncode': 0}
            with patch.object(cohort_experiments, 'make_jobs', return_value=(jobs, settings)), \
                 patch.object(cohort_experiments, 'load_volume', return_value={}), \
                 patch.object(cohort_experiments, 'validate_case'), \
                 patch.object(cohort_experiments.GPUWorker, 'run', side_effect=run):
                with self.assertRaises(SystemExit) as exit:
                    cohort_experiments.main(['--run', '--gpus', '0', '--output', str(output)])
                self.assertEqual(exit.exception.code, 1)
            self.assertEqual(visited, ['pair1', 'pair2', 'pair3'])
            result = json.loads((output / 'summary.json').read_text())['datasets']['acdc']
            self.assertEqual((result['intended'], result['completed'], result['successful']), (3, 3, 2))
            self.assertEqual(result['metrics']['pooled_label_DSC']['observations'], 6)
            self.assertEqual(result['pcg_converged'], 2)

    def test_frozen_acdc_directions_and_manifest_tampering(self):
        from medical.cohort_data import definition
        manifest = dict(protocol='acdc_kan_idir_v1', source_revision='3ac12e200159811eafb6192f3f8db3afef2a9d1a',
                        spacing=[1.5, 1.5, 3.15], subjects=[], pairs=[])
        for patient in range(101, 151):
            for phase in ('ED', 'ES'):
                name = f'patient{patient}_{phase}'
                manifest['subjects'].append(dict(subject=f'patient{patient}', phase=phase,
                                                 image=f'images/{name}.nii.gz', label=f'labels/{name}.nii.gz'))
        for a, b in [('ED', 'ES'), ('ES', 'ED')]:
            for patient in range(101, 151):
                moving, fixed = f'patient{patient}_{a}', f'patient{patient}_{b}'
                manifest['pairs'].append(dict(pair_index=len(manifest['pairs']) + 1, moving=moving, fixed=fixed,
                    moving_image=f'images/{moving}.nii.gz', moving_label=f'labels/{moving}.nii.gz',
                    fixed_image=f'images/{fixed}.nii.gz', fixed_label=f'labels/{fixed}.nii.gz'))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'manifest.json'
            path.write_text(json.dumps(manifest))
            result = definition(root, 'acdc')
            self.assertEqual(result['pairs'][0]['moving'], 'patient101_ED')
            self.assertEqual(result['pairs'][50]['moving'], 'patient101_ES')
            manifest['pairs'].reverse()
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                definition(root, 'acdc')


class TestNeuriteScoring(unittest.TestCase):
    def test_identity_uses_all_35_labels_on_release_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shape = (8, 9, 10)
            codes = (np.arange(np.prod(shape)).reshape(shape) % 36).astype(np.uint8)
            for name in ('fixed', 'moving'):
                nib.save(nib.Nifti1Image(codes, np.eye(4)), root / f'{name}.nii.gz')
            pair = {f'{k}_label': f'{k}.nii.gz' for k in ('moving', 'fixed')}
            mapping = identity_grid(shape[::-1], 'cpu').permute(2, 1, 0, 3)
            scores = evaluate({}, mapping, 'neurite_oasis', root, pair)
            self.assertEqual(scores['labels'], list(range(1, 36)))
            self.assertEqual(scores['DSC']['mean'], 1.)
            self.assertEqual(scores['HD95']['mean'], 0.)
            self.assertEqual(scores['LogJacDetStd']['mean'], 0.)
            self.assertEqual(scores['folding_fraction'], 0.)
