# numpy-mkl-ilp64

[![Build wheels](https://github.com/michael-denyer/numpy-mkl/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/michael-denyer/numpy-mkl/actions/workflows/build_wheels.yml)

**ILP64 Fork** - This fork builds NumPy with MKL's **ILP64 (64-bit integer)** interface,
enabling eigendecomposition of matrices larger than 46k x 46k (the LP64 limit).

> **Upstream**: [urob/numpy-mkl](https://github.com/urob/numpy-mkl) (LP64 version)

## Why ILP64?

MKL's default LP64 interface uses 32-bit integers for array indexing, limiting matrices to ~2.1 billion
elements (46k x 46k). For large-scale eigendecomposition (50k-200k samples), ILP64's 64-bit integers
remove this limitation.

| Interface | Integer Size | Max Matrix Elements | Max Samples (square) |
|-----------|--------------|---------------------|----------------------|
| LP64      | 32-bit       | ~2.1 billion        | ~46,000              |
| **ILP64** | **64-bit**   | **~9 quintillion**  | **~3 billion**       |

## Installation

```bash
# pip
pip install numpy --extra-index-url https://michael-denyer.github.io/numpy-mkl --force-reinstall --upgrade

# uv
uv add numpy --index https://michael-denyer.github.io/numpy-mkl
```

## Platform Support

| Platform | Runner         | BLAS Config            | Wheel Repair |
|----------|----------------|------------------------|--------------|
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

**Windows** - uses mkl-sdl (same as upstream), ILP64 verified via numerical tests:

```text
-Csetup-args=-Dblas=mkl-sdl
-Csetup-args=-Dlapack=mkl-sdl
-Csetup-args=--vsenv
```

## Compatibility Notes

- **ILP64 wheels are NOT compatible with LP64 wheels** - don't mix them
- Performance is similar to LP64 for most operations
- Windows wheels require the `mkl` pip package for runtime DLLs

## Original README

See [urob/numpy-mkl](https://github.com/urob/numpy-mkl) for full documentation on:
- Cross-platform collaborations
- Alternatives (Anaconda, Intel Distribution)
- Technical build details
