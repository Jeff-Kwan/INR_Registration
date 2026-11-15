"""Pinned upstream imports. No optimizer/model construction at import time."""
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SOURCES = json.loads((ROOT / 'sources.json').read_text())


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True)


def verify_source(name):
    spec = SOURCES[name]
    path = ROOT / 'upstream' / name
    if not (path / '.git').is_dir():
        raise RuntimeError(f'Missing {name} source; run /venv/main/bin/python -m baselines.setup')
    if path.is_symlink():
        raise RuntimeError(f'Refusing symlinked upstream checkout: {path}')
    if git(path, 'rev-parse', 'HEAD').strip() != spec['revision']:
        raise RuntimeError(f'{name}: upstream revision differs from sources.json')
    expected = (ROOT / spec['patch']).read_text() if spec.get('patch') else ''
    if git(path, 'diff', '--binary', 'HEAD') != expected:
        raise RuntimeError(f'{name}: source changes differ from the recorded compatibility patch')
    return path


def provenance(name):
    spec = SOURCES[name]
    return {k: spec[k] for k in ('url', 'revision', 'entry', 'patch') if k in spec}


@contextmanager
def upstream_imports(name):
    """Scope legacy top-level module names to one upstream checkout per process.

    Keep the context open during native execution. Independent cases run in
    separate processes; concurrently importing different methods is unsupported.
    """
    path = verify_source(name)
    search = [path, path / 'src'] if (path / 'src').is_dir() else [path]
    roots = {p.stem for base in search for p in base.iterdir()
             if p.suffix == '.py' or p.is_dir() and p.name != '.git'}
    previous_path = sys.path[:]
    previous_modules = {k: v for k, v in sys.modules.items() if k.split('.')[0] in roots}
    for key in previous_modules:
        del sys.modules[key]
    bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(p) for p in search]
    importlib.invalidate_caches()
    try:
        yield path
    finally:
        sys.path[:] = previous_path
        sys.dont_write_bytecode = bytecode
        for key in list(sys.modules):
            if key.split('.')[0] in roots:
                del sys.modules[key]
        sys.modules.update(previous_modules)


def native_modules(name):
    """Import only; do not instantiate a model or execute a registration."""
    imports = {
        'fireants': ['fireants.io.image', 'fireants.registration.syn'],
        'convexadam': ['convexAdam.convex_adam_MIND'],
        'ccidir': ['models.models', 'utils.general'],
        'nir': ['dataio', 'modules', 'efficient_modules', 'deform_net', 'training'],
    }
    modules = [importlib.import_module(module) for module in imports[name]]
    path = ROOT / 'upstream' / name
    if any(not Path(module.__file__).resolve().is_relative_to(path.resolve()) for module in modules):
        raise RuntimeError(f'{name}: imported a module outside the pinned checkout')
    return modules
