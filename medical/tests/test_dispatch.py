"""Shared-queue scheduling checks with mocked OASIS workers."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from medical.oasis_experiments import main


class TestDispatch(unittest.TestCase):
    def test_quick4_is_frozen_and_plan_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'not_created'
            stream = io.StringIO()
            with patch('medical.oasis_experiments.GPUWorker') as worker, redirect_stdout(stream):
                main(['--methods', 'sirenpath', '--subset', 'quick4', '--output', str(output)])
            plan = json.loads(stream.getvalue())
            self.assertEqual([job['pair_index'] for job in plan['jobs']], [1, 4, 8, 9])
            self.assertEqual(plan['subset']['dataset'], 'oasis')
            self.assertEqual(plan['subset']['selection_seed'], 42)
            self.assertFalse(plan['execution'])
            self.assertFalse(output.exists())
            worker.assert_not_called()

    def test_free_worker_takes_next_case_and_failure_does_not_stop_queue(self):
        first_started, third_started = threading.Event(), threading.Event()
        assigned = {}

        def run(worker, command, log):
            Path(log).touch()
            case = int(Path(command[command.index('--output') + 1]).name[4:])
            assigned[case] = worker.gpu
            if case == 1:
                first_started.set()
                if not third_started.wait(5):
                    raise AssertionError('Case 3 was stuck behind case 1')
            elif case == 2:
                if not first_started.wait(5):
                    raise AssertionError('First worker did not start')
            elif case == 3:
                third_started.set()
            return {'returncode': 1 if case == 2 else 0}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'input.json'
            config.write_text('{}')
            output = root / 'results'
            jobs = [
                {'method': 'ours', 'pair_index': i, 'seed': 42, 'output': str(output / f'case{i:02d}'),
                 'command': ['python', '--config', str(config), '--output', str(output / f'case{i:02d}')],
                 'missing_inputs': []}
                for i in range(1, 11)
            ]
            with (patch('medical.oasis_experiments.make_jobs', return_value=jobs),
                  patch('medical.gpu_worker.GPUWorker.run', autospec=True, side_effect=run),
                  redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error):
                main(['--methods', 'ours', '--gpus', '0', '1', '--output', str(output), '--run'])
            self.assertEqual(error.exception.code, 1)
            records = json.loads((output / 'summary.json').read_text())['jobs']
            self.assertEqual(sorted(record['pair_index'] for record in records), list(range(1, 11)))
            self.assertNotEqual(assigned[1], assigned[3])
            self.assertEqual(sum(record['returncode'] != 0 for record in records), 1)
            self.assertEqual(len(list(output.glob('case*.log'))), 10)
            self.assertEqual((output / 'configs/input.json').read_text(), config.read_text())


if __name__ == '__main__':
    unittest.main()
