"""Canonical model with the common official-comparison evaluator."""
from medical.comparison_metrics import dataset_for


def run_case(case, config, device, *, checkpoint_callback=None):
    dataset = dataset_for(case)
    if dataset in ('ixi', 'lpba40'):
        from medical.brain_method import run_case as run
    else:
        from medical.cohort_method import run_case as run
    return run(case, config, device, checkpoint_callback=checkpoint_callback, standardized=True)
