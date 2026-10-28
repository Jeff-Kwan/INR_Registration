"""Exercise sweep control flow with a fake clock and a fake trainer process."""
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from medical.experiments import hyperparameter_sweep as sweep


class SweepTests(unittest.TestCase):
    def test_launch_failure_is_recorded_without_starting_a_trainer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'sweep'
            base = sweep.load_config(sweep.ROOT / 'configs/lpba_pdlk.json')
            with patch.object(sweep, 'REPO_ROOT', root), patch.object(sys, 'argv', ['sweep', '--output', str(output)]), \
                 patch.object(sweep, 'design', return_value=[base]), \
                 patch.object(sweep.subprocess, 'Popen', side_effect=OSError('simulated launch failure')):
                sweep.main()
            records = json.loads((output / 'summary/trials.json').read_text())
            self.assertTrue(records)
            self.assertTrue(all(r['status'] == 'failed' for r in records))
            self.assertTrue(all('simulated launch failure' in r['error'] for r in records))

    def simulate(self, cutoff=None):
        clock = [1000.]
        launched = []

        class FakeTrainer:
            def wait(self, timeout):
                clock[0] += min(300., timeout)
                return 0

        def launch(command, **kwargs):
            self.assertEqual(command[1:3], ['-m', 'medical.registration.train'])
            config_path = Path(command[command.index('--config') + 1])
            config = json.loads(config_path.read_text())
            launched.append((config_path.parent.name.split('_', 1)[1], config, clock[0]))
            return FakeTrainer()

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            output = root / 'sweep'
            argv = ['sweep', '--output', str(output)]
            if cutoff:
                argv += ['--deadline-utc', cutoff]
            stack.enter_context(patch.object(sys, 'argv', argv))
            stack.enter_context(patch.object(sweep, 'REPO_ROOT', root))
            stack.enter_context(patch.object(sweep.time, 'monotonic', side_effect=lambda: clock[0]))
            stack.enter_context(patch.object(sweep.time, 'time', return_value=1000.))
            stack.enter_context(patch.object(sweep, 'estimate', return_value=300.))
            stack.enter_context(patch.object(sweep.subprocess, 'Popen', side_effect=launch))
            stack.enter_context(patch.object(sweep, 'measure', return_value={
                'dice': .7, 'hd95_mm': 6., 'subject_metrics': []}))
            sweep.main()
            overview = json.loads((output / 'summary/overview.json').read_text())
            self.assertEqual(overview['status'], 'finished')
            self.assertEqual(overview['attempts'], len(launched))
        return launched, clock[0] - 1000.

    def test_panels_controls_retention_and_scientific_deduplication(self):
        launched, elapsed = self.simulate()
        self.assertLessEqual(elapsed, 14400.)
        stages = {stage for stage, _, _ in launched}
        self.assertEqual(stages, {'screen', 'confirm', 'full', 'reserve_repeat'})
        for stage, config, _ in launched:
            self.assertEqual(config['evaluation']['min_endpoint_psnr_db'], 25.)
            self.assertEqual(config['recovery']['tolerance'], 2e-5)
            self.assertFalse(config['output']['save_volumes'])
            self.assertFalse(config['output']['save_frames'])
            self.assertFalse(config['output']['save_velocities'])
            self.assertEqual(config['output']['save_checkpoints'], stage not in ('screen', 'confirm'))
        for stage in ('confirm', 'full', 'reserve_repeat'):
            configs = [c for s, c, _ in launched if s == stage]
            self.assertEqual([c['lk']['affine_rho'] for c in configs[:2]], [0., .4])
            keys = [sweep.config_key(c) for c in configs]
            self.assertEqual(len(keys), len(set(keys)), f'{stage}: repeated scientific configuration')

    def test_replacement_cutoff_shortens_the_total_window(self):
        # The fake wall clock starts at 1000 seconds; cutoff is 2700 seconds.
        launched, elapsed = self.simulate('1970-01-01T00:45:00+00:00')
        self.assertTrue(launched)
        self.assertLessEqual(elapsed, 1700.)

    def test_duplicate_lock_exits_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'must not exist'
            with (root / '.relaxed_pdlk_sweep.lock').open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(sweep, 'REPO_ROOT', root), patch.object(sys, 'argv', ['sweep', '--output', str(output)]), \
                     patch.object(sweep.subprocess, 'Popen', side_effect=AssertionError('trainer launched')):
                    with self.assertRaises(SystemExit) as result:
                        sweep.main()
                self.assertEqual(result.exception.code, 1)
                self.assertFalse(output.exists())
