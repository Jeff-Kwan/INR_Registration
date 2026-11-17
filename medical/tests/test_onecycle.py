"""Pure scalar scheduler checks; no training or optimizer steps."""
import math
import unittest
from pathlib import Path

from medical.configuration import load_config, validate_config
from medical.registration.optimization import learning_rate_factor


class TestOneCycle(unittest.TestCase):
    def config(self):
        return load_config(Path(__file__).parents[1]/'configs/tuning/15_onecycle.json')

    def test_matches_historical_schedule_every_update(self):
        settings=self.config()['training']
        expected=[]
        for step in range(1,5001):
            fraction=(step-1)/4999
            if fraction<.1:
                rise=.5*(1-math.cos(math.pi*(fraction/.1)))
                factor=.25+.75*rise
            else:
                factor=.25+.75*.5*(1+math.cos(math.pi*((fraction-.1)/.9)))
            expected.append(factor)
            self.assertEqual(learning_rate_factor(settings,step),factor)
        self.assertAlmostEqual(expected[0],.25)
        self.assertAlmostEqual(expected[-1],.25)
        peak=max(range(5000),key=expected.__getitem__)
        self.assertEqual(peak+1,501)
        self.assertTrue(all(a<=b for a,b in zip(expected[:peak],expected[1:peak+1])))
        self.assertTrue(all(a>=b for a,b in zip(expected[peak:],expected[peak+1:])))
        self.assertAlmostEqual(sum(expected)*settings['lr'],.3124625)

    def test_only_lr_recipe_changes_and_default_is_unchanged(self):
        actual=self.config();default=load_config()
        for key in ('lr_schedule','lr_final_factor'):
            actual['training'][key]=default['training'][key]
        self.assertEqual(actual,default)
        self.assertEqual(default['training']['lr_schedule'],{'name':'cosine'})
        self.assertEqual(default['training']['lr_final_factor'],.2)

    def test_invalid_onecycle_options_are_rejected(self):
        for key,values in [('warmup_fraction',[0,1,-.1,float('nan'),True]),('start_factor',[0,1.1,float('inf'),True])]:
            for value in values:
                config=self.config();config['training']['lr_schedule'][key]=value
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):validate_config(config)
        config=self.config();config['training']['lr_schedule']['cycle_momentum']=True
        with self.assertRaises(ValueError):validate_config(config)
