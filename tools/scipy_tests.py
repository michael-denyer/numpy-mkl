"""ILP64 validation tests for scipy built against MKL ILP64.

Runs a focused linalg subset with timeout, not the full scipy.test() which
can be long/flaky on manylinux.
"""
import os
import sys

os.environ.setdefault('RUNNER_OS', 'Linux')

import scipy  # noqa: E402
import numpy as np  # noqa: E402

np.show_config()
print(f"scipy {scipy.__version__}")

# Quick smoke test: linalg eigendecomp
print("\n=== SciPy ILP64 smoke test ===\n")
M = np.random.randn(100, 100)
M = M @ M.T
eigenvalues, eigenvectors = scipy.linalg.eigh(M)
reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
error = np.abs(M - reconstructed).max()
print(f"  eigh reconstruction error: {error:.2e}")
assert error < 1e-8, f"Reconstruction error {error:.2e} too large"
print("  PASSED\n")

# Run scipy's linalg tests with timeout
print("Running scipy linalg test subset...")
passed = scipy.test(verbose=2, extra_argv=[
    '-x',
    '--timeout=120',
    '-k', 'linalg and not slow',
])

sys.exit(not passed)
