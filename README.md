# numpy-mkl-ilp64

[![Build wheels](https://github.com/michael-denyer/numpy-mkl/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/michael-denyer/numpy-mkl/actions/workflows/build_wheels.yml)

**ILP64 Fork** - This fork builds NumPy with MKL's **ILP64 (64-bit integer)** interface,
enabling eigendecomposition of matrices larger than 46k x 46k (the LP64 limit).

> **Upstream**: [urob/numpy-mkl](https://github.com/urob/numpy-mkl) (LP64 version)

This repository provides binary wheels for NumPy and SciPy, linked to Intel's high-performance
[oneAPI Math Kernel
Library](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) for Intel CPUs.
The wheels are accessible through a custom Python Package Index (PyPI) and can be installed with
`pip` or `uv`.

## Why ILP64?

MKL's default LP64 interface uses 32-bit integers for array indexing, limiting matrices to ~2.1 billion
elements (46k x 46k). For large-scale eigendecomposition (50k-200k samples), ILP64's 64-bit integers
remove this limitation.

| Interface | Integer Size | Max Matrix Elements | Max Samples (square) |
|-----------|--------------|---------------------|----------------------|
| LP64      | 32-bit       | ~2.1 billion        | ~46,000              |
| **ILP64** | **64-bit**   | **~9 quintillion**  | **~3 billion**       |

## Installation

MKL-accelerated wheels are available for 64-bit Linux and Windows. If using one of the
recommended package managers below, there are no other prerequisites; all dependencies are
automatically installed by the package manager.

```bash
# pip
pip install numpy scipy --index-url https://michael-denyer.github.io/numpy-mkl

# uv
uv add numpy --index https://michael-denyer.github.io/numpy-mkl
```

## Platform Support

| Platform | Runner         | BLAS Config            | Wheel Repair |
|----------|----------------|-------------------------|--------------|
| Linux    | manylinux_2_28 | mkl-dynamic-ilp64-iomp | auditwheel   |
| Windows  | windows-2022   | mkl-sdl                | delvewheel   |

## Build Changes from Upstream

**Linux** - explicit ILP64 flags with symbol suffix to force 64-bit integer resolution:

```diff
- -Csetup-args=-Dblas=mkl-sdl
- -Csetup-args=-Dlapack=mkl-sdl
+ -Csetup-args=-Dblas=mkl-dynamic-ilp64-iomp
+ -Csetup-args=-Dlapack=mkl-dynamic-ilp64-iomp
+ -Csetup-args=-Duse-ilp64=true
+ -Csetup-args=-Dblas-symbol-suffix=_64
```

**Windows** - uses mkl-sdl (same as upstream, LP64):

```text
-Csetup-args=-Dblas=mkl-sdl
-Csetup-args=-Dlapack=mkl-sdl
-Csetup-args=--vsenv
```

Only NumPy builds ILP64, and only on Linux. SciPy and mkl-service, on both platforms, and
NumPy on Windows, stay LP64. Mixing ILP64 NumPy with LP64 SciPy in the same environment is
untested; treat the two as separate deployment targets until proven otherwise.

## Compatibility Notes

- `numpy` wheels are compiled against a lower architecture bound of `X86_V3` (ca. 2013+ for
  Core and Xeon CPUs, ca. 2022+ for Atom CPUs). Older architectures are not supported.
- **ILP64 wheels are NOT compatible with LP64 wheels** - don't mix them.
- Performance is similar to LP64 for most operations.
- Wheels do not vendor MKL on either platform. They depend on the `mkl` runtime package, which
  ships the ILP64 interface library and the compute kernels MKL loads at runtime, and on the
  patched `mkl-service` from this index, which locates and preloads them on import.

## Original README

See [urob/numpy-mkl](https://github.com/urob/numpy-mkl) for full documentation on:
- Cross-platform collaborations
- Alternatives (Anaconda, Intel Distribution)
- Technical build details
