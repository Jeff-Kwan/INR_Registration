"""CPU-only figure/selection fixtures; no recovery or registration."""
import tempfile
from pathlib import Path
import json
import unittest
import numpy as np
from scripts.create_case_media import choose_slice,select_cases,save_prediction,plot_case


class TestCaseMedia(unittest.TestCase):
    def test_class_count_not_volume_and_centre_tie(self):
        labels=np.ones((8,9,5),np.uint8)
        labels[:,:,1]=0;labels[0,0,1]=1;labels[1,1,1]=2
        labels[:,:,3]=labels[:,:,1]
        index,counts=choose_slice(labels)
        self.assertEqual(index,1);self.assertEqual(counts[1],2)
        labels[2,2,2]=3;labels[1,1,2]=2
        self.assertEqual(choose_slice(labels)[0],2)

    def test_average_ratios_and_lower_median(self):
        rows=[dict(dataset='synthetic',pair_index=i,classical_advantage=a,neural_advantage=b) for i,a,b in [(1,-.5,.1),(2,0,0),(3,.3,.1),(4,.5,.5)]]
        selected=select_cases(dict(datasets=['synthetic'],rows=rows))
        self.assertEqual([r['pair_index'] for r in selected],[1,2,4])

    def test_identity_warp_saved_and_plotting(self):
        shape=(8,9,5);image=np.arange(np.prod(shape),dtype=np.float32).reshape(shape)
        labels=(np.indices(shape)[0]%3+1).astype(np.uint8)
        case={k:dict(data=image,affine=np.eye(4)) for k in ('fixed','moving')}
        grid=np.indices(shape).transpose(1,2,3,0).astype(np.float32)
        with tempfile.TemporaryDirectory() as scratch:
            root=Path(scratch)
            for method in ('a','b'):
                save_prediction(grid,root/method,case,labels,labels,dict(labels=[1,2,3],Dice=dict(mean=1.)),'synthetic identity')
                self.assertTrue(json.loads((root/method/'recovery.json').read_text())['matches_recorded_within_1e_4'])
            plot_case(root,case,labels,labels,['a','b'],dict(dataset='synthetic',rank='median',pair_index=2,average_advantage=0.),media_output=root/'media')
            self.assertTrue((root/'media/fixed_image.png').is_file());self.assertTrue((root/'media/a/warped_labels_overlay.png').is_file())

            self.assertTrue(all(p.suffix=='.png' for p in (root/'media').rglob('*') if p.is_file()))
