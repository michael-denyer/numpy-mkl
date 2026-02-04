# numpy-mkl-ilp64

**ILP64 Fork** - This fork builds NumPy and SciPy with MKL's **ILP64 (64-bit integer)** interface,
enabling eigendecomposition of matrices larger than 46k × 46k (the LP64 limit).

> **Upstream**: [urob/numpy-mkl](https://github.com/urob/numpy-mkl) (LP64 version)

## Why ILP64?

MKL's default LP64 interface uses 32-bit integers for array indexing, limiting matrices to ~2.1 billion
elements (46k × 46k). For large-scale eigendecomposition (50k-200k samples), ILP64's 64-bit integers
remove this limitation.

| Interface | Integer Size | Max Matrix Elements | Max Samples (square) |
|-----------|--------------|---------------------|----------------------|
| LP64      | 32-bit       | ~2.1 billion        | ~46,000              |
| **ILP64** | **64-bit**   | **~9 quintillion**  | **~3 billion**       |

## Installation

```bash
# Databricks / pip
pip install numpy scipy --extra-index-url https://michael-denyer.github.io/numpy-mkl --force-reinstall --upgrade

# uv
uv add numpy scipy --index https://michael-denyer.github.io/numpy-mkl
```

## Compatibility Notes

- **ILP64 wheels are NOT compatible with LP64 wheels** - don't mix them
- All packages (numpy, scipy, mkl-service) must use the same interface
- Performance is similar to LP64 for most operations

## Build Changes from Upstream

```diff
- -Csetup-args=-Dblas=mkl-sdl
- -Csetup-args=-Dlapack=mkl-sdl
+ -Csetup-args=-Dblas=mkl-dynamic-ilp64-seq
+ -Csetup-args=-Dlapack=mkl-dynamic-ilp64-seq
+ -Csetup-args=-Duse-ilp64=true
```

## Original README

See [urob/numpy-mkl](https://github.com/urob/numpy-mkl) for full documentation on:
- Cross-platform collaborations
- Alternatives (Anaconda, Intel Distribution)
- Technical build details
