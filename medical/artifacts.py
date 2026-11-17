"""Compact run artifacts and bounded-lifetime scratch. Never writes result maps."""
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile

import torch

FORMAT_VERSION = 2


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def fresh_directory(path):
    path = Path(path)
    protected = (Path(__file__).resolve().parents[1] / 'Data', Path.cwd() / 'Data')
    if any(path.resolve().is_relative_to(root.resolve()) for root in protected):
        raise ValueError('Result directories cannot be created under Data/')
    if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
        raise FileExistsError(f'Output must be a new or empty directory: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_output(command):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    return Path.cwd() / 'results' / command / stamp


def default_scratch():
    return Path.cwd() / 'results' / '.scratch'


def code_provenance():
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root))
        return {'revision': revision, 'dirty': dirty}
    except (OSError, subprocess.CalledProcessError):
        return {'revision': None, 'dirty': None}


def save_checkpoint(path, pair, config, metadata):
    payload = {'format_version': FORMAT_VERSION, 'dimensions': pair.dimensions,
               'model': pair.model_config, 'config': config, 'metadata': metadata,
               'state': {k: v.detach().cpu().clone() for k, v in pair.state_dict().items()}}
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device='cpu'):
    from medical.models.siren import PairedSiren
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format_version') != FORMAT_VERSION or set(payload) != {
            'format_version', 'dimensions', 'model', 'config', 'metadata', 'state'}:
        raise ValueError('Unsupported checkpoint; historical formats are archival only')
    if 'working_grid' in payload['config'] or payload['config'].get('spatial_units', 'voxel') != 'voxel':
        raise ValueError('Only native-voxel checkpoints are supported')
    pair = PairedSiren(payload['dimensions'], **payload['model']).to(device)
    pair.load_state_dict(payload['state'], strict=True)
    return pair.eval(), payload


@contextmanager
def training_log(output):
    with (Path(output) / 'training.csv').open('w', newline='') as handle:
        writer = None
        def report(row):
            nonlocal writer
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
        yield report


def cleanup_abandoned(root):
    """Remove only our marked scratch owned by a dead process on this host."""
    for path in Path(root).glob('recovery-*'):
        marker = path / 'owner.json'
        if path.is_symlink() or not marker.is_file() or marker.is_symlink():
            continue
        try:
            owner = json.loads(marker.read_text())
            if owner.get('host') != socket.gethostname() or type(owner.get('pid')) is not int or owner['pid'] < 1:
                continue
            os.kill(owner['pid'], 0)
        except ProcessLookupError:
            shutil.rmtree(path)
        except (OSError, ValueError):
            continue


@contextmanager
def recovery_scratch(root):
    root = Path(root)
    if root.is_symlink():
        raise ValueError('Scratch root must not be a symlink')
    root.mkdir(parents=True, exist_ok=True)
    cleanup_abandoned(root)
    with tempfile.TemporaryDirectory(prefix='recovery-', dir=root) as directory:
        path = Path(directory)
        write_json(path / 'owner.json', {'pid': os.getpid(), 'host': socket.gethostname()})
        yield path
