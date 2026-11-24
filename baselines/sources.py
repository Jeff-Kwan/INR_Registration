"""Pinned upstream imports. No optimizer/model construction at import time."""
from contextlib import contextmanager
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SOURCES = json.loads((ROOT / 'sources.json').read_text())


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True)


def trimmed_sources(name, path):
    """Recreate declared file trims from the exact pinned upstream blobs."""
    spec = SOURCES[name]
    for relative, ranges in spec.get('omit_lines', {}).items():
        target = path / relative
        if target.is_symlink() or not target.resolve().is_relative_to(path.resolve()):
            raise RuntimeError(f'{name}: invalid source path for trim: {relative}')
        blob = subprocess.check_output(['git', '-C', str(path), 'show', f"{spec['revision']}:{relative}"])
        lines = blob.splitlines(keepends=True)
        omitted = set()
        for first, last in ranges:
            if not 1 <= first <= last <= len(lines):
                raise RuntimeError(f'{name}: invalid source trim in {relative}')
            omitted.update(range(first, last + 1))
        yield target, b''.join(line for number, line in enumerate(lines, 1) if number not in omitted)


def verify_source(name):
    spec = SOURCES[name]
    if 'package' in spec:
        actual = importlib.metadata.version(spec['package'])
        if actual != spec['revision']:
            raise RuntimeError(f"Expected {spec['package']}=={spec['revision']}; found {actual}")
        return ROOT
    path = ROOT / 'upstream' / name
    if not (path / '.git').is_dir():
        raise RuntimeError(f'Missing {name} source; run /venv/main/bin/python -m baselines.setup')
    if path.is_symlink():
        raise RuntimeError(f'Refusing symlinked upstream checkout: {path}')
    if git(path, 'rev-parse', 'HEAD').strip() != spec['revision']:
        raise RuntimeError(f'{name}: upstream revision differs from sources.json')
    expected = (ROOT / spec['patch']).read_text() if spec.get('patch') else ''
    omitted_paths = []
    for target, trimmed in trimmed_sources(name, path):
        if not target.is_file() or target.read_bytes() != trimmed:
            raise RuntimeError(f'{name}: source differs from the recorded trim: {target.relative_to(path)}')
        omitted_paths.append(f':(exclude){target.relative_to(path).as_posix()}')
    if git(path, 'diff', '--binary', 'HEAD', '--', *omitted_paths) != expected:
        raise RuntimeError(f'{name}: source changes differ from the recorded compatibility patch')
    return path


def provenance(name):
    spec = SOURCES[name]
    return {k: spec[k] for k in ('url', 'revision', 'entry', 'patch', 'omit_lines') if k in spec}


@contextmanager
def upstream_imports(name):
    """Scope upstream top-level module names to one checkout per process.

    Keep the context open during native execution. Independent cases run in
    separate processes; concurrently importing different methods is unsupported.
    """
    path = verify_source(name)
    if 'package' in SOURCES[name] or name == 'niftyreg':
        yield path
        return
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


def niftyreg_binary(name):
    path = ROOT.parent / '.local/niftyreg-build/reg-apps' / name
    if not path.is_file():
        raise RuntimeError('Build NiftyReg with python -m baselines.setup --methods niftyreg')
    return path


def native_modules(name):
    """Import only; do not instantiate a model or execute a registration."""
    if name == 'demons':
        return [importlib.import_module('SimpleITK')]
    if name == 'niftyreg':
        binaries = [niftyreg_binary(n) for n in ('reg_f3d', 'reg_transform')]
        for binary in binaries:
            subprocess.run([str(binary), '--version'], check=True, capture_output=True)
        return binaries
    imports = {
        'nodeo': ['Registration', 'Network', 'Utils'],
        'dnvf': ['models.dnvf.networks.siren', 'data.datasets', 'utils.utils'],
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
