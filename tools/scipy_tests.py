import os
import sys


def finish_test_process(passed, runner_os):
    if runner_os == 'Linux':
        # The full suite can leave MKL/OpenMP state that races with extension
        # destructors during interpreter finalization. Preserve pytest's result
        # and stop this disposable test process before that native teardown.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(int(not passed))
        return
    sys.exit(not passed)


def cossin_reconstructs():
    """Check that cossin factors are unitary and reproduce their input.

    Keeps the reconstruction asserts of SciPy's test_cossin, which the Linux
    -k filter skips along with its cross-path comparisons. Same seed, sizes
    and tolerance as the upstream test.

    Returns:
        True if every case passes; failures are printed.
    """
    # Lazy: test_branch_fixes loads this module without numpy or scipy.
    import numpy as np  # noqa: PLC0415
    from scipy.linalg import cossin  # noqa: PLC0415
    from scipy.stats import ortho_group, unitary_group  # noqa: PLC0415

    rng = np.random.default_rng(1708093570726217)
    passed = True
    for dtype, group in ((np.float64, ortho_group), (np.complex128, unitary_group)):
        for m, p, q in ((40, 12, 20), (100, 50, 50)):
            x = np.asarray(group.rvs(m, random_state=rng), dtype=dtype)
            blocks = [x[:p, :q], x[:p, q:], x[p:, :q], x[p:, q:]]
            atol = m * 1e3 * np.finfo(dtype).eps
            eye = np.eye(m)
            for swap_sign in (False, True):
                for source, args in (('matrix', (x, p, q)), ('blocks', (blocks,))):
                    u, cs, vh = cossin(*args, swap_sign=swap_sign)
                    errors = {
                        'x - u @ cs @ vh': np.abs(x - u @ cs @ vh).max(),
                        'u unitary': np.abs(u @ u.conj().T - eye).max(),
                        'vh unitary': np.abs(vh @ vh.conj().T - eye).max(),
                    }
                    for check, err in errors.items():
                        if not err <= atol:
                            passed = False
                            print(
                                f'cossin {dtype.__name__} m={m} p={p} q={q} '
                                f'{source} swap_sign={swap_sign}: {check} '
                                f'max error {err:.3e} > {atol:.3e}'
                            )
    print(f'cossin reconstruction: {"passed" if passed else "FAILED"}')
    return passed


# Guard required: on Windows multiprocessing.Pool uses 'spawn', which reimports
# __main__ in each worker. Without this guard the workers would re-execute
# scipy.test(), causing recursive test runs and a deadlock in test_pool.
if __name__ == '__main__':
    import scipy
    from scipy.linalg import blas

    runner_os = os.environ.get('RUNNER_OS')
    expect_ilp64_raw = os.environ.get('EXPECT_SCIPY_ILP64', '').lower()
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
        f'EXPECT_SCIPY_ILP64 must be true or false, got {expect_ilp64_raw!r}',
    )
    require(blas.HAS_LP64, f'{runner_os} SciPy must expose LP64 BLAS')
    require(
        blas.HAS_ILP64 is expect_ilp64,
        f'{runner_os} SciPy ILP64={blas.HAS_ILP64}, expected {expect_ilp64}',
    )
    require(
        blas.get_blas_funcs('gemm', ilp64=False).int_dtype.name == 'int32',
        f'{runner_os} SciPy LP64 BLAS must use int32',
    )
    if expect_ilp64:
        require(
            blas.get_blas_funcs('gemm', ilp64=True).int_dtype.name == 'int64',
            f'{runner_os} SciPy ILP64 BLAS must use int64',
        )

    # Exclude tests with known MKL-specific numerical precision failures.
    # Use -k (name-based filter) rather than --deselect (path-based): passing
    # --deselect with a scipy/ path triggers early path resolution by pytest,
    # which initializes MKL threads before test_pool's fork() and deadlocks.
    extra_args = []

    match os.environ.get('RUNNER_OS'):
        case 'Windows':
            # test_gh22705: j0 for large arguments; MKL intercepts sin/cos at
            # runtime and its range reduction for x=1e15/1e30 doesn't meet the
            # rtol=5e-15 tolerance.
            # TestSphHarm: sph_harm_y_all and sph_harm_y disagree by up to 3e-4
            # relative on values of order 1e-20, where the two recurrences and
            # MKL's sin/cos cancel differently. Filter on the class rather than
            # test_all, which as a substring would match unrelated tests.
            extra_args += ['-k', 'not test_gh22705 and not TestSphHarm']
        case 'Linux':
            # test_support_moments_sample: MKL computes a Normal moment as exact
            # 0.0 vs ~3e-9, just exceeding the atol=2e-09 tolerance.
            # The rest fail since MKL 2026.2.0 on some runners: they compare two
            # code paths for the same problem, and MKL no longer returns
            # identical results across them. cossin: singular vectors come
            # back with flipped signs depending on compute_u/compute_vh/separate
            # (both valid; the reconstruction checks pass). rgi cubic: the
            # iterative spline solve lands ~1e-6 apart. descending_points_nd,
            # expm_cond: 1 ulp under exact-equality asserts. A '[' suffix
            # matches only the parametrized test, not test_cossin_error_* or
            # test_nonscalar_values_2.
            skips = [
                'test_support_moments_sample',
                'test_cossin[',
                'test_cossin_separate',
                '(test_decomp_cossin and test_properties)',
                'test_nonscalar_values[cubic]',
                'test_descending_points_nd',
                '(TestBatch and test_expm_cond)',
            ]
            extra_args += ['-k', ' and '.join(f'not {s}' for s in skips)]

    passed = scipy.test(extra_argv=extra_args or None)
    # After the suite: generating the inputs initializes MKL threads, which
    # must not happen before test_pool forks (see -k note above).
    passed = cossin_reconstructs() and passed
    finish_test_process(passed, runner_os)
