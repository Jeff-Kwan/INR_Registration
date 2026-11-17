"""Fetch pinned source or check imports, without running any method."""
import argparse
import json
import subprocess

from .sources import ROOT, SOURCES, native_modules, upstream_imports, verify_source


def fetch(name):
    spec = SOURCES[name]
    if 'package' in spec:
        verify_source(name)
        return
    path = ROOT / 'upstream' / name
    if path.exists():
        verify_source(name)
        if name == 'niftyreg':
            build_niftyreg(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    command = ['git', 'clone', '--depth', '1', '--filter=blob:none', '--no-checkout']
    if 'tag' in spec:
        command += ['--branch', spec['tag']]
    subprocess.run([*command, spec['url'], str(path)], check=True)
    if subprocess.run(['git', '-C', str(path), 'cat-file', '-e', spec['revision'] + '^{commit}'],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        subprocess.run(['git', '-C', str(path), 'fetch', '--depth', '1', 'origin', spec['revision']], check=True)
    if name != 'niftyreg':
        subprocess.run(['git', '-C', str(path), 'sparse-checkout', 'set', '--no-cone', '--stdin'],
                       input='\n'.join('/' + p for p in spec['paths']) + '\n', text=True, check=True)
    subprocess.run(['git', '-C', str(path), 'checkout', '--detach', spec['revision']], check=True)
    if spec.get('patch'):
        subprocess.run(['git', '-C', str(path), 'apply', str(ROOT / spec['patch'])], check=True)
    verify_source(name)
    if name == 'niftyreg':
        build_niftyreg(path)


def build_niftyreg(path):
    build = ROOT.parent / '.local/niftyreg-build'
    subprocess.run(['cmake', '-S', str(path), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                    '-DBUILD_TESTING=OFF', '-DUSE_CUDA=OFF', '-DUSE_OPENMP=ON'], check=True)
    subprocess.run(['cmake', '--build', str(build), '--target', 'reg_f3d', 'reg_transform', '-j', '4'], check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify sources and imports; do not download or run a method')
    parser.add_argument('--methods', nargs='+', choices=SOURCES, default=list(SOURCES))
    args = parser.parse_args(argv)
    status = {}
    for name in args.methods:
        try:
            if not args.check:
                fetch(name)
            with upstream_imports(name):
                native_modules(name)
            status[name] = {'source': 'verified', 'imports': 'ok', 'revision': SOURCES[name]['revision']}
        except Exception as exc:
            status[name] = {'error': f'{type(exc).__name__}: {exc}'}
    print(json.dumps(status, indent=2))
    if any('error' in value for value in status.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
