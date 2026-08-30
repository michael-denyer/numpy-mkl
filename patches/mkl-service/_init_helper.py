# Drop-in replacement for mkl-service's own import-time hook.
#
# Upstream only registers MKL's DLL directory, and only for Windows venvs (it
# checks for pyvenv.cfg and assumes '{sys.exec_prefix}/Library/bin'), so it is a
# no-op on Linux, and for global and --user installs. Instead, locate the installed
# 'mkl' package through its metadata, which works regardless of where it ended up.
#
# This runs before 'mkl/__init__.py' imports the '_mklinit' extension, which is
# the only ordering requirement.

import ctypes
import os
from importlib.metadata import PackageNotFoundError, files

MKL_DISTRIBUTION = 'mkl'
WINDOWS_RUNTIMES = (
    ('dispatcher', '*mkl_rt*.dll'),
    ('ILP64 interface', '*mkl_intel_ilp64*.dll'),
)
LINUX_RUNTIMES = (
    ('dispatcher', '*libmkl_rt.so*'),
    ('core', '*libmkl_core.so*'),
    ('threaded runtime', '*libmkl_intel_thread.so*'),
    ('sequential runtime', '*libmkl_sequential.so*'),
    ('LP64 interface', '*libmkl_intel_lp64.so*'),
    ('ILP64 interface', '*libmkl_intel_ilp64.so*'),
)


def _distribution_files(distribution=MKL_DISTRIBUTION):
    try:
        installed_files = files(distribution)
    except PackageNotFoundError as e:
        raise ImportError(
            f"Cannot initialize MKL runtime: distribution '{distribution}' "
            'is not installed'
        ) from e

    if installed_files is None:
        raise ImportError(
            f"Cannot initialize MKL runtime: distribution '{distribution}' "
            'has no installed file metadata'
        )
    return tuple(installed_files)


def _required_files(installed_files, required, distribution=MKL_DISTRIBUTION):
    located = []
    for role, pattern in required:
        match = next((path for path in installed_files if path.match(pattern)), None)
        if match is None:
            raise ImportError(
                f"Cannot initialize MKL runtime from distribution '{distribution}': "
                f"missing {role} library matching '{pattern}'"
            )

        try:
            path = match.locate().resolve(strict=True)
        except (AttributeError, FileNotFoundError, OSError, TypeError) as e:
            raise ImportError(
                f"Cannot initialize MKL runtime from distribution '{distribution}': "
                f"{role} library '{match}' is not an installed file"
            ) from e
        located.append((role, path))
    return tuple(located)


def _initialize_windows(installed_files):
    runtimes = _required_files(installed_files, WINDOWS_RUNTIMES)
    runtime_directories = {runtime.parent for _, runtime in runtimes}
    if len(runtime_directories) != 1:
        raise ImportError(
            f"Cannot initialize MKL runtime from distribution '{MKL_DISTRIBUTION}': "
            f'required DLLs span multiple directories: {sorted(map(str, runtime_directories))}'
        )
    runtime_directory = runtime_directories.pop()
    try:
        return os.add_dll_directory(runtime_directory)
    except OSError as e:
        raise ImportError(
            f"Cannot initialize MKL runtime from distribution '{MKL_DISTRIBUTION}': "
            f"failed to register DLL directory '{runtime_directory}': {e}"
        ) from e


def _initialize_linux(installed_files):
    mode = os.RTLD_LAZY | ctypes.RTLD_GLOBAL
    handles = []
    for role, runtime in _required_files(installed_files, LINUX_RUNTIMES):
        try:
            handles.append(ctypes.CDLL(runtime, mode=mode))
        except OSError as e:
            raise ImportError(
                f"Cannot initialize MKL runtime from distribution '{MKL_DISTRIBUTION}': "
                f"failed to load {role} library '{runtime}': {e}"
            ) from e
    return tuple(handles)


def initialize_mkl_runtime(platform=None):
    platform = os.name if platform is None else platform
    installed_files = _distribution_files()
    if platform == 'nt':
        return _initialize_windows(installed_files)
    if platform == 'posix':
        return _initialize_linux(installed_files)
    raise ImportError(
        f"MKL runtime initialization does not support platform '{platform}'"
    )


# The returned DLL-directory or CDLL handles must live as long as the process.
_runtime_handle = initialize_mkl_runtime()
