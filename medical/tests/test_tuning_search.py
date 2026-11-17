"""Synthetic result fixtures and scalar schedules only; never fit models."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from medical.brain_data import sha256
from medical.configuration import load_config,validate_config
from medical.registration.optimization import pdlk_weight,registration_weight
from medical import tuning_search as search


def fixture(root,row,value=.8,*,directory=None):
    directory=Path(directory) if directory is not None else Path(root)/'runs'/row['id'];directory.mkdir(parents=True)
    config=directory/'config.json';config.write_text(json.dumps(row['config']))
    jobs=[]
    for dataset in search.DATASETS:
        for index in range(1,33):
            target=directory/dataset/f'pair{index:03d}';target.mkdir(parents=True)
            scores={key:dict(mean=v) for key,v in dict(Dice=value,HD95=2.,NPJ=.1,SDlogJ=.2).items()}
            scores.update(split='development_tuning32',dataset=dataset,pair_index=index,seed=42,metric_protocol='standardized_v1')
            (target/'metrics.json').write_text(json.dumps(scores))
            jobs.append(dict(dataset=dataset,pair_index=index,output=str(target),returncode=0,seconds=1.))
    (directory/'plan.json').write_text(json.dumps(dict(config_sha256=sha256(config),tuning_manifest_sha256='cohort',split='development_tuning32',total_jobs=128,jobs=jobs)))
    (directory/'summary.json').write_text(json.dumps(dict(complete=True,jobs=jobs)))


class TestTuningSearch(unittest.TestCase):
    def catalog(self):
        return dict(candidates=search.candidates(load_config()),tuning_manifest_sha256='cohort')

    def test_twenty_unique_one_dimensional_candidates_and_schedule_endpoints(self):
        catalog=self.catalog();search.validate_catalog(catalog)
        self.assertEqual(len(catalog['candidates']),20)
        self.assertEqual(set(row['category'] for row in catalog['candidates'][1:]),set(search.CATEGORIES))
        config=catalog['candidates'][1]['config'];validate_config(config)
        self.assertAlmostEqual(pdlk_weight(config['objective'],1),.3)
        self.assertAlmostEqual(pdlk_weight(config['objective'],5000),.1)
        config['objective']['weight']=0
        self.assertEqual(pdlk_weight(config['objective'],1),0)
        for row in catalog['candidates']:
            if row['category']=='Registration weight schedule':
                c=row['config'];s=c['training'];w=c['objective']['registration']['weight']
                self.assertAlmostEqual(registration_weight(s,w,1),s['registration_weight_schedule']['start_weight'])
                self.assertAlmostEqual(registration_weight(s,w,5000),w)
        catalog['candidates'][1]['config']['model']['width']=256
        with self.assertRaises(ValueError):search.validate_catalog(catalog)

    def test_ranking_includes_default_assembles_independent_winners_and_rejects_partial(self):
        catalog=self.catalog()
        with tempfile.TemporaryDirectory() as scratch:
            root=Path(scratch)
            for row in catalog['candidates']:fixture(root,row,.8 if row['category'] is None else .81)
            report=search.rank(catalog,root)
            self.assertEqual(set(report['winners']),set(search.CATEGORIES))
            expected=copy.deepcopy(catalog['candidates'][0]['config'])
            for category,paths in search.CATEGORIES.items():
                winner=next(r for r in catalog['candidates'] if r['id']==report['winners'][category])
                for path in paths:search.put(expected,path,search.get(winner['config'],path))
            self.assertEqual(report['combined_config'],expected)
            search.write_ranking(root,report)
            self.assertTrue((root/'RANKING.md').is_file())
            target=root/'runs'/'20'/'ixi'/'pair032'/'metrics.json';target.unlink()
            with self.assertRaises(ValueError):search.rank(catalog,root)

    def test_exact_ties_keep_default(self):
        catalog=self.catalog()
        with tempfile.TemporaryDirectory() as scratch:
            for row in catalog['candidates']:fixture(scratch,row)
            report=search.rank(catalog,scratch)
            self.assertTrue(all(v=='01_default' for v in report['winners'].values()))
            self.assertEqual(report['combined_config'],load_config())

    def test_provenance_mismatch_and_duplicate_case_rejected(self):
        row=self.catalog()['candidates'][0]
        with tempfile.TemporaryDirectory() as scratch:
            fixture(scratch,row);root=Path(scratch)/'runs'/row['id']
            with self.assertRaises(ValueError):search.read_scores(root,row['config'],'different-cohort')
            summary=json.loads((root/'summary.json').read_text());summary['jobs'][-1]=summary['jobs'][0]
            (root/'summary.json').write_text(json.dumps(summary))
            with self.assertRaises(ValueError):search.read_scores(root,row['config'],'cohort')

    def test_pipeline_combination_does_not_promote_or_launch_official_on_tie(self):
        catalog=self.catalog();catalog['default_sha256']=sha256(search.ROOT/'medical/configs/default.json')
        with tempfile.TemporaryDirectory() as scratch:
            root=Path(scratch);(root/'catalog.json').write_text(json.dumps(catalog))
            (root/'configs').mkdir()
            for row in catalog['candidates']:(root/'configs'/(row['id']+'.json')).write_text(json.dumps(row['config']))
            launches=[]
            def fake_launch(config,output,gpus):
                launches.append(output)
                fixture(None,dict(config=json.loads(config.read_text())),directory=output)
            with patch.object(search,'launch',side_effect=fake_launch),patch.object(search,'promote') as promote,patch.object(search.subprocess,'run') as official:
                search.pipeline(root,[0])
            self.assertEqual(len(launches),21)
            promote.assert_not_called();official.assert_not_called()
            self.assertFalse(json.loads((root/'decision.json').read_text())['promote'])

    def test_promotion_document_rendering_uses_active_values(self):
        config=load_config();config['objective']['c']=8
        text=search.render_method_document(config)
        self.assertIn('Promoted active scalar recipe',text)
        self.assertNotIn('**c=4**',text)
        self.assertNotIn('The canonical default remains cosine',text)

    def test_three_of_four_rule_and_missing_hd95(self):
        def scores(dice,hd):
            return dict(macro_Dice=sum(dice)/4,datasets={d:dict(Dice=dict(mean=x,scored_pairs=32),HD95=dict(mean=y,scored_pairs=32)) for d,x,y in zip(search.DATASETS,dice,hd)})
        base=scores([.8]*4,[2.]*4)
        candidate=scores([.81,.81,.81,.7],[1.9,1.9,2.1,1.9])
        result=search.compare_to_default(candidate,base)
        self.assertTrue(result['qualifies']) # The three datasets can differ.
        candidate['datasets']['ixi']['HD95']['scored_pairs']=31
        self.assertFalse(search.compare_to_default(candidate,base)['qualifies'])
        self.assertFalse(search.compare_to_default(base,base)['qualifies'])

    def test_final_pool_can_promote_category_winner_over_combination(self):
        def scores(value,hd):
            return dict(macro_Dice=value,datasets={d:dict(Dice=dict(mean=value,scored_pairs=32),HD95=dict(mean=hd,scored_pairs=32)) for d in search.DATASETS})
        report=dict(scores={'01_default':scores(.8,2.),'02':scores(.81,1.9)},winners={c:'02' for c in search.CATEGORIES})
        result=search.select_final(report,scores(.79,2.1))
        self.assertEqual(result['selected'],'02');self.assertTrue(result['promote'])

    def test_promotion_updates_default_and_documentation_in_temporary_workspace(self):
        original=search.ROOT
        config=load_config();config['objective']['c']=8
        decision=dict(selected='10',default_scores=dict(macro_Dice=.8),selected_scores=dict(macro_Dice=.81))
        with tempfile.TemporaryDirectory() as scratch:
            root=Path(scratch);(root/'medical/configs').mkdir(parents=True);(root/'docs').mkdir();(root/'search').mkdir()
            for relative in ('medical/configs/default.json','docs/METHOD.md','docs/TUNING_COHORT.md','README.md','AGENTS.md'):
                (root/relative).write_text((original/relative).read_text())
            catalog=dict(default_sha256=sha256(root/'medical/configs/default.json'))
            with patch.object(search,'ROOT',root):search.promote(root/'search',catalog,config,decision)
            self.assertEqual(json.loads((root/'medical/configs/default.json').read_text()),config)
            self.assertTrue((root/'search/previous_default.json').is_file())
            self.assertTrue((root/'docs/TUNING_PROMOTION.md').is_file())
            self.assertIn('promoted values',(root/'AGENTS.md').read_text())
