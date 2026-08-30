#!/usr/bin/env python

import argparse
import os
import zipfile
from dataclasses import dataclass
from email.parser import BytesParser
from importlib import metadata
from pathlib import Path


def normalize_distribution(name):
    return name.lower().replace('_', '-').replace('.', '-')


@dataclass(frozen=True)
class WheelIdentity:
    distribution: str
    version: str
    path: Path

    @classmethod
    def read(cls, expected_distribution, path):
        path = path.resolve(strict=True)
        with zipfile.ZipFile(path) as wheel:
            metadata_names = [
                name for name in wheel.namelist() if name.endswith('.dist-info/METADATA')
            ]
            if len(metadata_names) != 1:
                raise AssertionError(
                    f'{path} contains {len(metadata_names)} distribution metadata files'
                )
            message = BytesParser().parsebytes(wheel.read(metadata_names[0]))

        distribution = message['Name']
        version = message['Version']
        if normalize_distribution(distribution) != normalize_distribution(
            expected_distribution
        ):
            raise AssertionError(
                f'{path} contains {distribution!r}, expected {expected_distribution!r}'
            )
        return cls(distribution, version, path)

    def assert_installed(self):
        installed_version = metadata.version(self.distribution)
        if installed_version != self.version:
            raise AssertionError(
                f'{self.distribution} {installed_version} is installed, but same-run wheel '
                f'{self.path.name} contains {self.version}'
            )


def verify_package_set(wheels, runner_os):
    identities = tuple(
        WheelIdentity.read(distribution, path) for distribution, path in wheels.items()
    )
    for identity in identities:
        identity.assert_installed()

    import mkl  # noqa: F401, PLC0415
    import numpy as np  # noqa: PLC0415
    import scipy  # noqa: PLC0415
    from numpy.testing import HAS_LAPACK64  # noqa: PLC0415
    from scipy.linalg import blas  # noqa: PLC0415

    config = np.show_config(mode='dicts')
    blas_name = config.get('Build Dependencies', {}).get('blas', {}).get('name', '')
    lapack_name = config.get('Build Dependencies', {}).get('lapack', {}).get('name', '')
    assert 'mkl' in blas_name.lower(), f'NumPy BLAS must be MKL, got {blas_name}'
    assert 'mkl' in lapack_name.lower(), f'NumPy LAPACK must be MKL, got {lapack_name}'
    assert HAS_LAPACK64, f'{runner_os} NumPy wheel must use ILP64'

    assert blas.HAS_LP64, f'{runner_os} SciPy wheel must expose LP64 BLAS'
    if runner_os == 'Linux':
        assert blas.HAS_ILP64, 'Linux SciPy wheel must expose ILP64 BLAS'
        assert blas.get_blas_funcs('gemm', ilp64=True).int_dtype.name == 'int64'
    elif runner_os == 'Windows':
        assert not blas.HAS_ILP64, 'Windows SciPy wheel must remain LP64'
    else:
        raise AssertionError(f'Unexpected runner OS {runner_os!r}')
    assert blas.get_blas_funcs('gemm', ilp64=False).int_dtype.name == 'int32'

    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.array([1.0, 2.0])
    np.testing.assert_allclose(matrix @ np.linalg.solve(matrix, rhs), rhs)
    np.testing.assert_allclose(matrix @ scipy.linalg.solve(matrix, rhs), rhs)


def main():
    parser = argparse.ArgumentParser(description='Verify one same-run ILP64 wheel set')
    parser.add_argument('--mkl-service', required=True, type=Path)
    parser.add_argument('--numpy', required=True, type=Path)
    parser.add_argument('--scipy', required=True, type=Path)
    parser.add_argument('--runner-os', default=os.environ.get('RUNNER_OS'))
    args = parser.parse_args()

    verify_package_set(
        {
            'mkl-service': args.mkl_service,
            'numpy': args.numpy,
            'scipy': args.scipy,
        },
        args.runner_os,
    )


if __name__ == '__main__':
    main()
