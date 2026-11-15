"""Planner and scheduler checks with subprocesses mocked; no registrations run."""
from collections import Counter
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from baselines.run_all import METHODS,REPO,ROOT,make_jobs,main,execute


class RunAllTests(unittest.TestCase):
    def test_exact_three_method_cohorts(self):
        jobs=make_jobs(list(METHODS),['dirlab','abdomenctct','oasis'],REPO/'Data',ROOT/'results'/'test_plan')
        self.assertEqual(len(jobs),222)
        self.assertEqual(len({j['output'] for j in jobs}),222)
        self.assertEqual(Counter(j['method'] for j in jobs),{name:74 for name in METHODS})
        for method in METHODS:
            self.assertEqual(Counter(j['dataset'] for j in jobs if j['method']==method),{'dirlab':10,'abdomenctct':45,'oasis':19})
        self.assertTrue(all(j['command'][2]=='baselines.case_runner' for j in jobs))
        self.assertFalse(any(j['method']=='ccidir' for j in jobs))

    def test_default_only_prints_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'not_created'
            with patch('baselines.run_all.make_jobs',return_value=[]),patch('baselines.run_all.execute') as launch,redirect_stdout(io.StringIO()) as stdout:
                main(['--output',str(output)])
            launch.assert_not_called()
            self.assertFalse(output.exists())
            plan=json.loads(stdout.getvalue())
            self.assertFalse(plan['execution']);self.assertEqual(plan['gpus'],[0,1,2,3])

    def test_default_results_inside_baselines(self):
        with patch('baselines.run_all.make_jobs',return_value=[]),patch('baselines.run_all.execute') as launch,redirect_stdout(io.StringIO()) as stdout:
            main(['--methods','nir'])
        self.assertTrue(Path(json.loads(stdout.getvalue())['output']).is_relative_to(ROOT/'results'))
        launch.assert_not_called()

    def test_ccidir_excluded_and_data_output_refused(self):
        with self.assertRaises(ValueError):
            make_jobs(['ccidir'],['dirlab'],REPO/'Data',ROOT/'results'/'bad')
        with redirect_stdout(io.StringIO()),patch('sys.stderr',io.StringIO()),self.assertRaises(SystemExit):
            main(['--output',str(REPO/'Data'/'bad')])

    def test_four_gpu_scheduler_and_failure_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'outputs'
            source=root/'native_config.json';source.write_text('{"seed":42}\n')
            jobs=[]
            for n in range(8):
                target=output/'dirlab'/'fireants'/f'case{n:02d}'
                jobs.append({'method':'fireants','dataset':'dirlab','case':n,'seed':42,'job_id':f'dirlab/fireants/{n}',
                             'output':str(target),'command':['python','-m','baselines.case_runner','--config',str(source),'--output',str(target)],'missing_inputs':[]})
            barrier=threading.Barrier(4)
            def fake_run(command,**kwargs):
                self.assertEqual(kwargs['env']['OMP_NUM_THREADS'],'4')
                self.assertIn(kwargs['env']['CUDA_VISIBLE_DEVICES'],['0','1','2','3'])
                self.assertEqual(kwargs['cwd'],REPO)
                config_path=Path(command[command.index('--config')+1])
                self.assertTrue(config_path.is_relative_to(output/'configs'))
                target=Path(command[command.index('--output')+1])
                barrier.wait(timeout=10)
                code=1 if target.name=='case07' else 0
                if not code:
                    target.mkdir()
                    (target/'metrics.json').write_text(json.dumps({'idir_rounded_tre_mm':{'mean_mm':1}}))
                return subprocess.CompletedProcess(command,code)
            with patch('baselines.run_all.verify_source'),patch('baselines.run_all.subprocess.run',side_effect=fake_run) as run,redirect_stdout(io.StringIO()):
                summary=execute(jobs,[0,1,2,3],output)
            self.assertEqual(run.call_count,8)
            self.assertEqual({j['gpu'] for j in summary['jobs']},{0,1,2,3})
            self.assertEqual(summary['failed_jobs'],1)
            self.assertEqual(summary['datasets']['dirlab']['fireants']['unscored_direction_results'],1)
            self.assertEqual(json.loads((output/'progress.json').read_text())['finished'],8)
            self.assertEqual(source.read_text(),'{"seed":42}\n')
            with self.assertRaises(FileExistsError),patch('baselines.run_all.verify_source'):
                execute(jobs,[0],output)

if __name__=='__main__':unittest.main()
