"""CPU-only checks for frozen development selection; no model fitting."""
import copy
import unittest
from scripts.prepare_tuning_cohort import select, validate_selection


class TestTuningSelection(unittest.TestCase):
    def test_reproducible_diverse_training_only_selection(self):
        rows=select()
        self.assertEqual(rows,select())
        self.assertNotEqual(rows,select(43))
        validate_selection(rows)
        self.assertEqual(sum(map(len,rows.values())),128)

    def test_reject_test_subject_and_reused_patient(self):
        for dataset in ('oasis','acdc','lpba40','ixi'):
            rows=copy.deepcopy(select())
            if dataset=='oasis':rows[dataset][0]['moving']='OASIS_0395'
            elif dataset=='acdc':rows[dataset][0]=copy.deepcopy(rows[dataset][1])
            elif dataset=='lpba40':rows[dataset][0]['moving']='S11'
            else:rows[dataset][0]['fixed']='not_an_author_training_subject'
            with self.subTest(dataset=dataset),self.assertRaises(ValueError):validate_selection(rows)
