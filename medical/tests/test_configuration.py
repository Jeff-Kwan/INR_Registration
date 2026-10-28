"""Saved-config compatibility checks; no medical images or checkpoints are opened."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from medical.registration.configuration import load_config
from medical.registration.train import DEFAULT_CONFIG, resolve_dataset_path, validate_config


class ConfigurationTests(unittest.TestCase):
    def test_saved_pointwise_paths_resolve_without_changing_saved_file(self):
        current = json.loads(DEFAULT_CONFIG.read_text())
        for prefix in ('Data/', '/workspace/INR/Data/'):
            legacy = copy.deepcopy(current)
            legacy['dataset'].pop('root')
            for name in ('fixed', 'moving_dir', 'label_dir', 'fixed_label'):
                legacy['dataset'][name] = prefix + 'LPBA40/LPBA40_delineation/' + current['dataset'][name]
            legacy['lk'].update(patch_size=11, patches=64)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'saved config.json'
                original = json.dumps(legacy)
                path.write_text(original)
                loaded = load_config(path)
                validate_config(loaded)
                self.assertEqual(path.read_text(), original)
                self.assertEqual(loaded['lk'], current['lk'])
                self.assertEqual(loaded['model'], current['model'])
                self.assertEqual(loaded['training'], current['training'])
                self.assertEqual(loaded['output'], current['output'])
                expected = resolve_dataset_path(current, current['dataset']['fixed']).resolve()
                actual = resolve_dataset_path(loaded, loaded['dataset']['fixed']).resolve()
                self.assertEqual(actual, expected)

    def test_legacy_patch_configs_are_not_reinterpreted_as_pointwise(self):
        legacy = json.loads(DEFAULT_CONFIG.read_text())
        legacy['dataset'].pop('root')
        for name in ('fixed', 'moving_dir', 'label_dir', 'fixed_label'):
            legacy['dataset'][name] = 'Data/' + legacy['dataset'][name]
        legacy['lk'].update(mode='patch', patch_size=11, patches=64)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'config.json'
            path.write_text(json.dumps(legacy))
            loaded = load_config(path)
            self.assertEqual(loaded['lk']['mode'], 'patch')
            self.assertIn('patch_size', loaded['lk'])
            with self.assertRaises(ValueError):
                validate_config(loaded)
