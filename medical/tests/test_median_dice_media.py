"""Pure selection/crop checks; no model recovery or fitting."""
import unittest
import numpy as np
from scripts.create_median_dice_media import selections,crop_bounds


class TestMedianDiceMedia(unittest.TestCase):
    def test_median_and_two_baselines_per_group(self):
        groups=dict(classical=['c1','c2','c3'],neural=['n1','n2','n3'])
        rows=[]
        for i,dice in enumerate([.9,.6,.8,.7],1):
            row=dict(dataset='test',pair_index=i,ours_dice=dice)
            row.update({m+'_dice':v for m,v in zip(groups['classical']+groups['neural'],[.5,.8,.7,.6,.9,.7])});rows.append(row)
        result=selections(dict(datasets=['test'],rows=rows,groups=groups))[0]
        self.assertEqual(result['pair_index'],4)
        self.assertEqual(result['methods'],['ours','c2','c3','n2','n3'])
        self.assertEqual(result['strongest_baseline'],'n2')

    def test_common_crop_covers_both_predictions_and_fixed(self):
        a=np.zeros((30,40),bool);b=a.copy();c=a.copy();a[10,10]=True;b[15,20]=True;c[25,30]=True
        self.assertEqual(crop_bounds([a,b,c],padding=2),(8,28,8,33))
        self.assertEqual(crop_bounds([np.zeros((3,4),bool)]),(0,3,0,4))

    def test_ten_percent_crop_growth_and_clamping(self):
        from scripts.create_structure_contours import expanded_crop,dice
        self.assertEqual(expanded_crop([10,30,20,40],(100,100)),[9,31,19,41])
        self.assertEqual(expanded_crop([0,20,0,20],(30,30)),[0,22,0,22])
        self.assertEqual(expanded_crop([0,30,0,30],(30,30)),[0,30,0,30])
        a=np.array([[True,False],[False,True]])
        self.assertEqual(dice(a,a),1.)
