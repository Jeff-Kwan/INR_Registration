"""Shared-queue scheduling check with mocked workers; no GPU work."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from scripts.run_dirlab_cases import main
from medical.abdomen_experiments import main as learn2reg_main


class TestDispatch(unittest.TestCase):
    def test_dirlab_quick4_is_smallest_and_plan_only(self):
        with tempfile.TemporaryDirectory() as td:
            output=Path(td)/'not_created';stream=io.StringIO()
            argv=['run_dirlab_cases','--subset','quick4','--plan-only','--output',str(output)]
            with patch('sys.argv',argv),patch('scripts.run_dirlab_cases.GPUWorker') as worker,redirect_stdout(stream):
                main()
            plan=json.loads(stream.getvalue())
            self.assertEqual([j['case'] for j in plan['jobs']],[1,3,4,5])
            sizes=plan['subset']['all_case_sizes']
            ranked=sorted(sizes,key=lambda r:(r['pair_voxel_count'],r['case']))
            self.assertEqual([r['case'] for r in ranked[:4]],[1,4,3,5])
            self.assertFalse(plan['execution']);self.assertFalse(output.exists())
            worker.assert_not_called()

    def test_quick4_is_frozen_and_plan_only(self):
        for dataset, indices in (('oasis',[1,4,8,9]),('abdomenctct',[2,8,18,41])):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as td:
                output=Path(td)/'not_created'
                stream=io.StringIO()
                with patch('medical.abdomen_experiments.GPUWorker') as worker, redirect_stdout(stream):
                    learn2reg_main(['--methods','ours','--subset','quick4','--output',str(output)],dataset=dataset)
                plan=json.loads(stream.getvalue())
                self.assertEqual([j['pair_index'] for j in plan['jobs']],indices)
                self.assertEqual(plan['subset']['dataset'],dataset)
                self.assertEqual(plan['subset']['selection_seed'],42)
                self.assertFalse(plan['execution']);self.assertFalse(output.exists())
                worker.assert_not_called()

    def test_free_worker_takes_next_case_and_failure_does_not_stop_queue(self):
        for dataset in ('dirlab','oasis','abdomenctct'):
            first_started, third_started = threading.Event(), threading.Event()
            assigned = {}
            def run(worker, command, log):
                Path(log).touch()
                case = int(Path(command[command.index('--output')+1]).name[4:])
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
                return {'returncode':1 if case == 2 else 0}
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = root/'input.json'; config.write_text('{}')
                output = root/'results'
                argv = ['run_dirlab_cases', '--config', str(config), '--output', str(output), '--gpus', '0', '1']
                jobs=[{'method':'ours','pair_index':i,'seed':42,'output':str(output/f'case{i:02d}'),
                       'command':['python','--config',str(config),'--output',str(output/f'case{i:02d}')],
                       'missing_inputs':[]} for i in range(1,11)]
                with (patch('medical.abdomen_experiments.make_jobs',return_value=jobs), patch('sys.argv', argv), patch('medical.gpu_worker.GPUWorker.run', autospec=True, side_effect=run),
                      redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error):
                    if dataset=='dirlab': main()
                    else: learn2reg_main(['--methods','ours','--gpus','0','1','--output',str(output),'--run'],dataset=dataset)
                self.assertEqual(error.exception.code, 1)
                records = json.loads((output/'summary.json').read_text())
                if isinstance(records,dict): records=records['jobs']
                self.assertEqual(sorted(r.get('case',r.get('pair_index')) for r in records), list(range(1,11)))
                self.assertNotEqual(assigned[1], assigned[3])
                self.assertEqual(sum(r['returncode'] != 0 for r in records), 1)
                self.assertEqual(len(list(output.glob('case*.log'))), 10)
                snapshot=output/'config.json' if dataset=='dirlab' else output/'configs/input.json'
                self.assertEqual(snapshot.read_text(), config.read_text())


if __name__ == '__main__':
    unittest.main()
