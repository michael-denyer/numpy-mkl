#!/usr/bin/env python

import ctypes
import os
import runpy
import tempfile
import unittest
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

HELPER = Path(__file__).parents[1] / 'patches/mkl-service/_init_helper.py'
LINUX_NAMES = (
    'libmkl_rt.so.3',
    'libmkl_core.so.3',
    'libmkl_intel_thread.so.3',
    'libmkl_sequential.so.3',
    'libmkl_intel_lp64.so.3',
    'libmkl_intel_ilp64.so.3',
)
WINDOWS_NAMES = ('mkl_rt.3.dll',)


@contextmanager
def linux_platform():
    with (
        patch.object(os, 'name', 'posix'),
        patch.object(os, 'RTLD_LAZY', 1, create=True),
        patch.object(ctypes, 'RTLD_GLOBAL', 256, create=True),
    ):
        yield


class PackageFile:
    def __init__(self, path):
        self.path = Path(path)

    def __str__(self):
        return self.path.name

    def match(self, pattern):
        return self.path.match(pattern)

    def locate(self):
        return self.path


class TestMklRuntimeInitialization(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.runtime_dir = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def runtime_files(self, names=LINUX_NAMES):
        paths = []
        for name in names:
            path = self.runtime_dir / name
            path.touch()
            paths.append(PackageFile(path))
        return paths

    @patch('importlib.metadata.files')
    @patch('ctypes.CDLL')
    def test_linux_loads_required_libraries_in_order_and_retains_handles(
        self, cdll, distribution_files
    ):
        distribution_files.return_value = list(reversed(self.runtime_files()))
        handles = [object() for _ in LINUX_NAMES]
        cdll.side_effect = handles

        with linux_platform():
            namespace = runpy.run_path(str(HELPER))

        mode = 1 | 256
        self.assertEqual(
            cdll.call_args_list,
            [
                call((self.runtime_dir / name).resolve(), mode=mode)
                for name in LINUX_NAMES
            ],
        )
        self.assertEqual(namespace['_runtime_handle'], tuple(handles))

    @patch('importlib.metadata.files', side_effect=PackageNotFoundError('mkl'))
    def test_missing_distribution_is_contextual_import_error(self, _distribution_files):
        with self.assertRaisesRegex(ImportError, "distribution 'mkl' is not installed"):
            runpy.run_path(str(HELPER))

    @patch('importlib.metadata.files', return_value=None)
    def test_missing_distribution_metadata_is_contextual_import_error(
        self, _distribution_files
    ):
        with self.assertRaisesRegex(ImportError, 'has no installed file metadata'):
            runpy.run_path(str(HELPER))

    @patch('importlib.metadata.files')
    def test_missing_required_library_names_role_and_pattern(self, distribution_files):
        distribution_files.return_value = self.runtime_files(LINUX_NAMES[:-1])

        with (
            linux_platform(),
            self.assertRaisesRegex(
                ImportError,
                "missing ILP64 interface library matching '\\*libmkl_intel_ilp64",
            ),
        ):
            runpy.run_path(str(HELPER))

    @patch('importlib.metadata.files')
    def test_missing_required_file_names_metadata_entry(self, distribution_files):
        installed_files = self.runtime_files()
        installed_files[-1].path.unlink()
        distribution_files.return_value = installed_files

        with (
            linux_platform(),
            self.assertRaisesRegex(ImportError, 'is not an installed file'),
        ):
            runpy.run_path(str(HELPER))

    @patch('importlib.metadata.files')
    @patch('ctypes.CDLL')
    def test_linux_loader_error_names_role_and_path(self, cdll, distribution_files):
        distribution_files.return_value = self.runtime_files()
        cdll.side_effect = OSError('wrong ELF class')

        with (
            linux_platform(),
            self.assertRaisesRegex(
                ImportError,
                "failed to load dispatcher library '.*libmkl_rt.so.3': wrong ELF",
            ),
        ):
            runpy.run_path(str(HELPER))

    @patch('importlib.metadata.files')
    def test_windows_retains_dll_directory_handle(self, distribution_files):
        distribution_files.return_value = self.runtime_files(WINDOWS_NAMES)
        directory_handle = object()
        runtime_handle = SimpleNamespace(DGEMM_64=object(), DGESV_64=object())
        with (
            patch.object(os, 'name', 'nt'),
            patch.object(
                os,
                'add_dll_directory',
                return_value=directory_handle,
                create=True,
            ) as add,
            patch.object(
                ctypes, 'WinDLL', return_value=runtime_handle, create=True
            ) as load,
        ):
            namespace = runpy.run_path(str(HELPER))

        add.assert_called_once_with(self.runtime_dir.resolve())
        load.assert_called_once_with((self.runtime_dir / WINDOWS_NAMES[0]).resolve())
        self.assertEqual(
            namespace['_runtime_handle'], (directory_handle, runtime_handle)
        )

    @patch('importlib.metadata.files', return_value=[])
    def test_windows_missing_runtime_is_contextual_import_error(
        self, _distribution_files
    ):
        with (
            patch.object(os, 'name', 'nt'),
            patch.object(os, 'add_dll_directory', create=True),
            self.assertRaisesRegex(
                ImportError, "missing dispatcher library matching '\\*mkl_rt\\*\\.dll'"
            ),
        ):
            runpy.run_path(str(HELPER))

    @patch('importlib.metadata.files')
    def test_windows_requires_ilp64_exports(self, distribution_files):
        distribution_files.return_value = self.runtime_files(('mkl_rt.3.dll',))
        with (
            patch.object(os, 'name', 'nt'),
            patch.object(os, 'add_dll_directory', create=True),
            patch.object(
                ctypes,
                'WinDLL',
                return_value=SimpleNamespace(DGEMM_64=object()),
                create=True,
            ),
            self.assertRaisesRegex(
                ImportError,
                "dispatcher.*missing ILP64 export 'DGESV_64'",
            ),
        ):
            runpy.run_path(str(HELPER))

    @patch('importlib.metadata.files')
    def test_windows_loader_error_names_dll_directory(self, distribution_files):
        distribution_files.return_value = self.runtime_files(WINDOWS_NAMES)
        with (
            patch.object(os, 'name', 'nt'),
            patch.object(
                os,
                'add_dll_directory',
                side_effect=OSError('access denied'),
                create=True,
            ),
            self.assertRaisesRegex(
                ImportError, 'failed to register DLL directory.*access denied'
            ),
        ):
            runpy.run_path(str(HELPER))


if __name__ == '__main__':
    unittest.main()
