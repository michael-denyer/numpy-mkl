import os
import sys
from pathlib import Path

if os.environ.get('RUNNER_OS') == 'Windows':
    mkl_bin = Path(sys.prefix) / 'Library' / 'bin'
    if mkl_bin.is_dir():
        os.add_dll_directory(str(mkl_bin))

    # GH 20391
    (Path(sys.prefix) / 'libs').mkdir(parents=True, exist_ok=True)

import numpy as np
from numpy.testing import HAS_LAPACK64

np.show_config()
config = np.show_config(mode='dicts')
blas_name = config.get('Build Dependencies', {}).get('blas', {}).get('name', '')
lapack_name = config.get('Build Dependencies', {}).get('lapack', {}).get('name', '')
runner_os = os.environ.get('RUNNER_OS', '')

assert runner_os in {'Linux', 'Windows'}, (
    f'RUNNER_OS must identify a configured build platform, got {runner_os!r}'
)
assert 'mkl' in blas_name.lower(), f'BLAS must be MKL, got {blas_name}'
assert 'mkl' in lapack_name.lower(), f'LAPACK must be MKL, got {lapack_name}'
assert HAS_LAPACK64, (
    f'{runner_os} NumPy wheel must use ILP64, but HAS_LAPACK64={HAS_LAPACK64} '
    f'and BLAS={blas_name}'
)

a = np.array([[4.0, 1.0], [1.0, 3.0]])
b = np.array([1.0, 2.0])
np.testing.assert_allclose(a @ np.linalg.solve(a, b), b)

os.environ['NPY_AVAILABLE_MEM'] = '4 GB'
version = tuple(map(int, np.version.version.split('.')))
if version >= (2, 2, 0):
    passed = np.test(label='full', extra_argv=['-n=auto', '--timeout=1800'])
else:
    passed = np.test(label='full', extra_argv=['-n=auto'])
sys.exit(not passed)
