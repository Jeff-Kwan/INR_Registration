"""No-training raw crossfade control, using the shared SIRENPath stage-3 solver."""
from medical.sirenpath_ablation_method import run_case as run_shared


def run_case(case, config, device):
    return run_shared(case, config, device, raw=True)
