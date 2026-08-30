import os
import sys
from pathlib import Path

if os.environ.get('RUNNER_OS') == 'Windows':
    # GH 20391
    (Path(sys.prefix) / 'libs').mkdir(parents=True, exist_ok=True)

import numpy as np
from numpy.testing import HAS_LAPACK64

np.show_config()
config = np.show_config(mode='dicts')
blas_name = config.get('Build Dependencies', {}).get('blas', {}).get('name', '')
lapack_name = config.get('Build Dependencies', {}).get('lapack', {}).get('name', '')
runner_os = os.environ.get('RUNNER_OS', '')
expect_ilp64_raw = os.environ.get('EXPECT_NUMPY_ILP64', '').lower()
expect_ilp64 = expect_ilp64_raw == 'true'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


require(
    runner_os in {'Linux', 'Windows'},
    f'RUNNER_OS must identify a configured build platform, got {runner_os!r}',
)
require(
    expect_ilp64_raw in {'true', 'false'},
    f'EXPECT_NUMPY_ILP64 must be true or false, got {expect_ilp64_raw!r}',
)
require('mkl' in blas_name.lower(), f'BLAS must be MKL, got {blas_name}')
require('mkl' in lapack_name.lower(), f'LAPACK must be MKL, got {lapack_name}')
require(
    HAS_LAPACK64 is expect_ilp64,
    f'{runner_os} NumPy ILP64={HAS_LAPACK64}, expected {expect_ilp64}; BLAS={blas_name}',
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
