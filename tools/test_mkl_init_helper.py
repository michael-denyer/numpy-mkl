#!/usr/bin/env python

import ctypes
import os
import runpy
import unittest
from pathlib import Path
from unittest.mock import call, patch

HELPER = Path(__file__).parents[1] / 'patches/mkl-service/_init_helper.py'


class PackageFile:
    def __init__(self, name):
        self.name = name

    def match(self, pattern):
        return Path(self.name).match(pattern)

    def locate(self):
        return Path('/prefix/lib') / self.name


@unittest.skipIf(os.name == 'nt', 'Linux preload test')
class TestLinuxPreload(unittest.TestCase):
    @patch('importlib.metadata.files')
    @patch('ctypes.CDLL')
    def test_loads_dispatcher_before_direct_ilp64_group(self, cdll, files):
        files.return_value = [
            PackageFile('libmkl_intel_ilp64.so.3'),
            PackageFile('libmkl_intel_lp64.so.3'),
            PackageFile('libmkl_intel_thread.so.3'),
            PackageFile('libmkl_sequential.so.3'),
            PackageFile('libmkl_rt.so.3'),
            PackageFile('libmkl_core.so.3'),
        ]

        namespace = runpy.run_path(str(HELPER))

        mode = os.RTLD_LAZY | ctypes.RTLD_GLOBAL
        expected_names = (
            'libmkl_rt.so.3',
            'libmkl_core.so.3',
            'libmkl_intel_thread.so.3',
            'libmkl_sequential.so.3',
            'libmkl_intel_lp64.so.3',
            'libmkl_intel_ilp64.so.3',
        )
        self.assertEqual(
            cdll.call_args_list,
            [
                call((Path('/prefix/lib') / name).resolve(), mode=mode)
                for name in expected_names
            ],
        )
        self.assertEqual(namespace['_preloaded_libraries'], [cdll.return_value] * 6)


if __name__ == '__main__':
    unittest.main()
