"""Planning must not spawn registration workers or create output directories."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from baselines.run_dirlab import main
from baselines.sources import native_modules, SOURCES, upstream_imports
from baselines.tests.test_adapters import config


class PlanningTests(unittest.TestCase):
    def test_default_is_plan_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            with patch('baselines.run_dirlab.verify_source'), patch('subprocess.run') as run, redirect_stdout(io.StringIO()) as text:
                main(['--cases', '1', '--output', str(output), '--data', directory])
            run.assert_not_called()
            self.assertFalse(output.exists())
            plan = json.loads(text.getvalue())
            self.assertFalse(plan['execution'])
            self.assertEqual(len(plan['jobs']), 53)
            self.assertTrue(all(job['missing_inputs'] for job in plan['jobs']))

    def test_presets_match_native_option_names(self):
        # Imports and signature/default inspection only. No constructors/fit/solve.
        import inspect
        from types import SimpleNamespace
        for name in SOURCES:
            with self.subTest(name=name), upstream_imports(name):
                modules = native_modules(name)
                native = config(name)['native']
                if name == 'fireants':
                    self.assertLessEqual(set(native), set(inspect.signature(modules[1].SyNRegistration).parameters))
                elif name == 'convexadam':
                    self.assertLessEqual(set(native), set(inspect.signature(modules[0].convex_adam_pt).parameters))
                elif name == 'ccidir':
                    dummy = SimpleNamespace()
                    modules[0].ImplicitRegistrator.set_default_arguments(dummy)
                    self.assertLessEqual(set(native), set(dummy.args))
                else:
                    parameters = inspect.signature(modules[-1].train_efficient_pairwise).parameters
                    self.assertIn('loss_weight', parameters)
                    # Check native normalization round-trips without loading images.
                    import numpy as np
                    points = np.array([[1.2, 2.3, 3.4]], dtype=np.float32)
                    normalized = modules[0].normalize_coords(points.copy(), (5, 7, 9), (5, 7, 9))
                    np.testing.assert_allclose(modules[0].unnormalize_coords(normalized, (5, 7, 9), (5, 7, 9)), points, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
