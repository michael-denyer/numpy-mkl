#!/usr/bin/env python

import argparse
import fnmatch
import sys
import zipfile
from pathlib import Path, PurePosixPath

# MKL and its OpenMP runtime are installed by the mkl and intel-openmp packages, and
# mkl-service preloads them. A copy vendored into <package>.libs is a second runtime
# in the same process: numpy 2.4.6 through 2.5.3 shipped libiomp5 there, and the two
# copies tore each other down in _dl_fini with an exit-time SIGSEGV.
FORBIDDEN = ('libiomp5*', 'libmkl*', 'libtbb*', 'mkl_*', 'tbb*')


def vendored_runtime_libraries(wheel):
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    vendored = []
    for name in names:
        parts = PurePosixPath(name).parts
        if len(parts) < 2 or not parts[0].endswith('.libs'):  # noqa: PLR2004
            continue
        if any(fnmatch.fnmatch(parts[-1].lower(), pattern) for pattern in FORBIDDEN):
            vendored.append(name)
    return vendored


def main():
    parser = argparse.ArgumentParser(
        description='Fail when a wheel vendors an MKL or OpenMP runtime library'
    )
    parser.add_argument('wheels', nargs='+', type=Path)
    args = parser.parse_args()

    failed = False
    for wheel in args.wheels:
        vendored = vendored_runtime_libraries(wheel)
        if vendored:
            failed = True
            print(f'::error::{wheel.name} vendors runtime libraries:')
            for name in vendored:
                print(f'  {name}')
        else:
            print(f'{wheel.name}: no vendored MKL or OpenMP runtime')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
