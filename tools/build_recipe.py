#!/usr/bin/env python

import argparse
import hashlib
from pathlib import Path

COMMON_FILES = (
    '.github/workflows/package_set.yml',
    '.github/workflows/verify_package_set.yml',
    '.github/workflows/wheels.yml',
    'ci-targets.yaml',
    'patches/LICENSE_MKL.txt',
    'pyproject.toml',
    'uv.lock',
    'tools/build_recipe.py',
    'tools/fetch_matrix2',
    'tools/get_file_in_pkg',
    'tools/package_set_plan.py',
    'tools/store_info.py',
    'tools/verify_package_set.py',
    'tools/write_build_info.py',
)

PACKAGE_FILES = {
    'mkl-service': (
        'patches/mkl-service/_init_helper.py',
        'tools/test_mkl_init_helper.py',
    ),
    'numpy': (
        'patches/numpy/cython_3_3_0_limited_api.patch',
        'patches/numpy/init_mkl.patch',
        'tools/numpy_tests.py',
    ),
    'scipy': (
        'patches/scipy/init_mkl.patch',
        'tools/scipy_tests.py',
    ),
}


def iter_files(package, root):
    for relative in (*COMMON_FILES, *PACKAGE_FILES[package]):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        yield path


def recipe_hash(package, root=None):
    root = (Path.cwd() if root is None else Path(root)).resolve()
    digest = hashlib.sha256()
    for path in iter_files(package, root):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b'\0')
        digest.update(path.read_bytes())
        digest.update(b'\0')
    return f'sha256:{digest.hexdigest()}'


def main():
    parser = argparse.ArgumentParser(description='Hash the local wheel build recipe')
    parser.add_argument('package', choices=sorted(PACKAGE_FILES))
    parser.add_argument('--root', type=Path, default=Path.cwd())

    args = parser.parse_args()
    print(recipe_hash(args.package, args.root))


if __name__ == '__main__':
    main()
