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
            extra_args += ['-k', 'not test_support_moments_sample']

    passed = scipy.test(extra_argv=extra_args or None)
    finish_test_process(passed, runner_os)
