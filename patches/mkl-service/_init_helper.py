# Drop-in replacement for mkl-service's own import-time hook.
#
# Upstream only registers MKL's DLL directory, and only for Windows venvs (it
# checks for pyvenv.cfg and assumes '{sys.exec_prefix}/Library/bin'), so it is a
# no-op on Linux, and for global and --user installs. Instead, locate the installed
# 'mkl' package through its metadata, which works regardless of where it ended up.
#
# This runs before 'mkl/__init__.py' imports the '_mklinit' extension, which is
# the only ordering requirement.

import contextlib
import ctypes
import os
from importlib.metadata import PackageNotFoundError, files

with contextlib.suppress(
    AttributeError, FileNotFoundError, PackageNotFoundError, StopIteration, TypeError
):
    if os.name == 'nt':
        # Add the MKL library path to the DLL search path.
        dll = next(p for p in files('mkl') if p.match('*mkl_rt*.dll'))
        os.add_dll_directory(dll.locate().resolve().parent)
    else:
        # The direct MKL libraries refer to symbols in each other without ELF
        # dependency edges, so the loader must see the group in this order with
        # lazy binding. Load libmkl_rt first so its LP64 and ILP64 entry points
        # take precedence over the direct ILP64 interface.
        mkl_files = tuple(files('mkl'))
        mode = os.RTLD_LAZY | ctypes.RTLD_GLOBAL
        patterns = (
            '*libmkl_rt.so*',
            '*libmkl_core.so*',
            '*libmkl_intel_thread.so*',
            '*libmkl_intel_ilp64.so*',
        )
        _preloaded_libraries = [
            ctypes.CDLL(
                next(p for p in mkl_files if p.match(pattern)).locate().resolve(),
                mode=mode,
            )
            for pattern in patterns
        ]
