"""Fitted crossfade of the uncalibrated endpoint reconstructions from run 5."""
from pathlib import Path
from medical.abdomen_evaluation import PROTOCOLS
from medical.sirenpath_ablation_method import run_case as run_shared


def run_case(case, config, device):
    settings = dict(config)
    root = Path(settings.pop('canonical_checkpoint_root'))
    names = tuple(Path(case[k]['path']).name for k in ('fixed','moving'))
    indices = [i for i,pair in enumerate(PROTOCOLS['oasis']['eval_pairs'],1)
               if tuple(Path(pair[k]).name for k in ('fixed','moving')) == names]
    if len(indices) != 1:
        raise ValueError('Fitted crossfade requires an official L2R-OASIS pair')
    checkpoint = root/f'pair{indices[0]:03d}'/'checkpoint.pt'
    return run_shared(case, settings, device, reference_checkpoint=checkpoint)
