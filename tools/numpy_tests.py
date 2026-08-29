"""ILP64 validation tests for numpy built against MKL ILP64.

Validates that:
1. MKL is the active BLAS backend
2. ILP64 (64-bit integer) interface is configured
3. BLAS/LAPACK operations produce correct results (not garbage from LP64 mismatch)
4. A focused subset of numpy's linalg tests pass
"""
import os
import sys
import traceback
from pathlib import Path

if os.environ.get('RUNNER_OS') == 'Windows':
    # Register MKL DLL directory before importing numpy. The mkl pip package
    # installs DLLs to {venv}/Library/bin/ (conda-style layout). Python 3.8+
    # doesn't search PATH for DLLs, so we must register explicitly.
    mkl_bin = Path(sys.prefix) / 'Library' / 'bin'
    if mkl_bin.is_dir():
        os.add_dll_directory(str(mkl_bin))

    # GH 20391
    libs = Path(sys.prefix) / 'libs'
    libs.mkdir(parents=True, exist_ok=True)

import numpy as np
from numpy.testing import HAS_LAPACK64

np.show_config()


# ---------------------------------------------------------------------------
# Gate: assert the wheel has the integer width this repo meant to build.
#
# HAS_LAPACK64 reads numpy.linalg._umath_linalg._ilp64, compiled into the
# extension at build time, so it reports what was actually linked. The
# pkg-config name in show_config() does not: mkl-sdl carries no 'ilp64' in its
# name, and MKL's LP64 libraries export the _64 symbols too, so a name match
# proves nothing either way. Nor do the numerical tests below: every matrix
# here is under 1000x1000, far short of the ~46k LP64 ceiling, so they pass
# identically on both builds.
# ---------------------------------------------------------------------------
print("\n=== Asserting ILP64 + MKL configuration ===\n")

config = np.show_config(mode='dicts')
blas_name = config.get('Build Dependencies', {}).get('blas', {}).get('name', '')
lapack_name = config.get('Build Dependencies', {}).get('lapack', {}).get('name', '')

print(f"  BLAS:        {blas_name}")
print(f"  LAPACK:      {lapack_name}")
print(f"  LAPACK ILP64: {HAS_LAPACK64}")

runner_os = os.environ.get('RUNNER_OS', '')
assert runner_os, "RUNNER_OS must be set; refusing to guess the expected integer width"

# Platforms this repo builds ILP64. Keep in step with the meson args in
# .github/workflows/wheels.yml.
ILP64_PLATFORMS = {'Linux'}
expect_ilp64 = runner_os in ILP64_PLATFORMS

assert 'mkl' in blas_name.lower(), f"BLAS must be MKL, got: {blas_name}"
assert 'mkl' in lapack_name.lower(), f"LAPACK must be MKL, got: {lapack_name}"

assert HAS_LAPACK64 == expect_ilp64, (
    f"{runner_os} wheel should be {'ILP64' if expect_ilp64 else 'LP64'}, "
    f"but numpy reports HAS_LAPACK64={HAS_LAPACK64} (blas={blas_name})"
)

print(f"  MKL active, {'ILP64' if expect_ilp64 else 'LP64'} as intended: CONFIRMED\n")


# ---------------------------------------------------------------------------
# Numerical validation tests
# ---------------------------------------------------------------------------
print("=== ILP64 Numerical Validation ===\n")

failures = 0


def run_test(name, fn):
    """Run a test, print result, count failures."""
    global failures
    print(f"Test: {name}...")
    sys.stdout.flush()
    try:
        fn()
        print("  PASSED")
    except Exception as e:
        failures += 1
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()


def test_basic_ops():
    a = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    b = np.array([5, 4, 3, 2, 1], dtype=np.float64)
    assert np.allclose(a + b, [6, 6, 6, 6, 6])
    assert np.allclose(np.dot(a, b), 35)


def test_matmul():
    A = np.random.randn(100, 100)
    B = np.random.randn(100, 100)
    C = A @ B
    assert C.shape == (100, 100)


def test_eigendecomp():
    M = np.random.randn(50, 50)
    M = M @ M.T
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    print(f"  eigenvalues: min={eigenvalues.min():.6g}, max={eigenvalues.max():.6g}")
    # Reconstruction must be tight -- garbage from LP64 mismatch gives error ~1e+1
    reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    error = np.abs(M - reconstructed).max()
    print(f"  reconstruction error: {error:.2e}")
    assert error < 1e-8, f"Reconstruction error {error:.2e} too large (LP64 mismatch?)"


def test_svd():
    X = np.random.randn(30, 20)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    reconstructed = U @ np.diag(s) @ Vt
    error = np.abs(X - reconstructed).max()
    print(f"  reconstruction error: {error:.2e}")
    assert error < 1e-10, f"SVD reconstruction error {error:.2e} too large"


def test_solve():
    A = np.random.randn(50, 50)
    A = A @ A.T + 50 * np.eye(50)
    b = np.random.randn(50)
    x = np.linalg.solve(A, b)
    error = np.abs(A @ x - b).max()
    print(f"  residual error: {error:.2e}")
    assert error < 1e-10, f"Solve residual {error:.2e} too large"


def test_large_eigendecomp():
    """1000x1000 -- still small vs 46k limit, but exercises the full path."""
    M = np.random.randn(1000, 1000)
    M = M @ M.T
    eigenvalues = np.linalg.eigvalsh(M)
    assert len(eigenvalues) == 1000
    # Eigenvalues of M @ M.T must be non-negative (within floating point)
    assert eigenvalues.min() > -1e-6, \
        f"Min eigenvalue {eigenvalues.min():.6g} is too negative"
    print(f"  min eigenvalue: {eigenvalues.min():.6g}")


run_test("Basic array operations", test_basic_ops)
run_test("Matrix multiplication (BLAS)", test_matmul)
run_test("Eigendecomposition (LAPACK)", test_eigendecomp)
run_test("SVD decomposition", test_svd)
run_test("Linear solve", test_solve)
run_test("1000x1000 eigendecomposition", test_large_eigendecomp)

if failures:
    print(f"\n=== {failures} test(s) FAILED ===\n")
    sys.exit(1)

print("\n=== All ILP64 validation tests PASSED ===\n")

# ---------------------------------------------------------------------------
# Run numpy's own linalg tests (focused subset, not full suite)
# ---------------------------------------------------------------------------
print("Running numpy linalg test subset...")
os.environ['NPY_AVAILABLE_MEM'] = '4 GB'

passed = np.test(label='fast', verbose=2, extra_argv=[
    '-x',               # Stop on first failure
    '--timeout=120',    # Per-test timeout
    '-k', 'linalg',    # Only linalg tests
])

sys.exit(not passed)
