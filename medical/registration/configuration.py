"""Read current and recognized saved pointwise configurations.

This module only reads JSON. It never modifies saved artifacts, opens image
files, loads models, or changes scientific settings.
"""
import json
from pathlib import Path


def load_config(path):
    config = json.loads(Path(path).read_text())
    dataset = config.get('dataset')
    if not isinstance(dataset, dict) or 'root' in dataset:
        return config

    # Recognize only documented saved path layouts; never guess provenance.
    for name in ('fixed', 'moving_dir', 'label_dir', 'fixed_label'):
        value = dataset.get(name)
        for prefix in ('Data/', '/workspace/INR/Data/'):
            if isinstance(value, str) and value.startswith(prefix):
                dataset[name] = value[len(prefix):]
                break
        else:
            raise ValueError(
                f'Legacy dataset.{name} has no known Data/ prefix; '
                'supply an explicit dataset.root and paths relative to it')
    dataset['root'] = '../Data'

    # These old patch-sampling fields were unused by pointwise trials. Preserve
    # real patch configs so the maintained validator rejects them explicitly.
    lk = config.get('lk', {})
    if lk.get('mode') == 'pointwise':
        lk.pop('patch_size', None)
        lk.pop('patches', None)
    return config
