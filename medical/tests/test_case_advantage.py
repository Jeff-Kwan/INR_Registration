"""Synthetic JSON fixtures only; never analyze real experiment results."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import rank_case_advantage as analysis


class TestCaseAdvantage(unittest.TestCase):
    def fixture(self,root,methods,values):
        jobs=[]
        for method in methods:
            for index,value in enumerate(values,1):
                target=root/'ixi'/method/f'pair{index:03d}';target.mkdir(parents=True)
                score=dict(dataset='ixi',metric_protocol='standardized_v1',labels=[1,2],Dice=dict(mean=value))
                (target/'metrics.json').write_text(json.dumps(score))
                jobs.append(dict(dataset='ixi',method=method,pair_index=index,output=str(target),returncode=0,
                                 command=['python','--moving',f'/synthetic/m{index}','--fixed',f'/synthetic/f{index}'],preparation_sha256={'protocol':'same'}))
        for name in ('plan.json','summary.json'):(root/name).write_text(json.dumps(dict(jobs=jobs)))

    def test_signed_advantage_and_zero_denominator(self):
        self.assertAlmostEqual(analysis.advantage(.8,[.5,.5,.5,.5])[1],.6)
        self.assertAlmostEqual(analysis.advantage(.4,[.5,.5,.5,.5])[1],-.2)
        with self.assertRaises(ValueError):analysis.advantage(.8,[0,0,0,0])

    def test_extrema_keep_ties_and_use_group_specific_case_means(self):
        with tempfile.TemporaryDirectory() as scratch,patch.dict(analysis.DATASETS,{'ixi':3},clear=True):
            root=Path(scratch);ours=root/'ours';classic=root/'classic';neural=root/'neural'
            self.fixture(ours,['ours'],[.8,.4,.8]);self.fixture(classic,analysis.GROUPS['classical'],[.5,.5,.5]);self.fixture(neural,analysis.GROUPS['neural'],[.8,.2,.8])
            result=analysis.analyze(ours,classic,neural,['ixi'])
            self.assertEqual([r['pair_index'] for r in result['extrema']['ixi']['classical']['highest']],[1,3])
            self.assertEqual(result['extrema']['ixi']['classical']['lowest'][0]['pair_index'],2)
            self.assertEqual(result['extrema']['ixi']['neural']['highest'][0]['pair_index'],2)
            self.assertAlmostEqual(result['rows'][0]['classical_advantage_percent'],60)
            analysis.write_report(result,root/'report');self.assertTrue((root/'report/REPORT.md').is_file())
            # A swapped direction must never be silently compared by pair index.
            p=neural/'plan.json';plan=json.loads(p.read_text());plan['jobs'][0]['command'][2]='/different/moving';p.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):analysis.analyze(ours,classic,neural,['ixi'])

    def test_failed_case_rejected_instead_of_reducing_baseline_mean(self):
        with tempfile.TemporaryDirectory() as scratch,patch.dict(analysis.DATASETS,{'ixi':1},clear=True):
            root=Path(scratch);self.fixture(root,['ours'],[.8]);p=root/'summary.json';data=json.loads(p.read_text());data['jobs'][0]['returncode']=1;p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):analysis.index_run(root,['ours'],['ixi'])
