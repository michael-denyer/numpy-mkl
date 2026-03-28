<!-- GSD:project-start source:PROJECT.md -->
## Project

**numpy-mkl: Windows Wheel Support**

A CI/CD pipeline that builds ILP64 NumPy and mkl-service wheels linked against Intel MKL, published to a GitHub Pages PyPI index. Currently Linux-only. This milestone adds Windows wheel builds.

**Core Value:** Ship ILP64 NumPy wheels for Windows that are as reliable as the existing Linux builds — correct MKL linkage, passing tests, bundled DLLs.

### Constraints

- **Platform**: Windows builds use `windows-latest` GitHub runner, no container
- **Compiler**: MSVC via Visual Studio (meson `--vsenv` flag)
- **DLL bundling**: Must use delvewheel (auditwheel is Linux-only)
- **pkg-config**: Not available by default on Windows, needs choco install
- **Compatibility**: Changes must not break existing Linux builds
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Python 3.11+ - Entire project (build tools, CI utilities, testing infrastructure)
## Runtime
- Python 3.11, 3.12, 3.13, 3.14 (multi-version support)
- Linux (manylinux_2_28), macOS, Windows (via CI matrix)
- `uv` ~0.9 - Fast Python package manager, replaces pip/poetry for development and builds
- Lockfile: `uv.lock` (present, committed)
## Frameworks
- Meson - Build system for NumPy and mkl-service compilation via `uv build --wheel`
- NumPy (numpy/numpy repo) - Primary package being built with MKL ILP64
- mkl-service (IntelPython/mkl-service repo) - MKL runtime service dependency
- auditwheel - Linux wheel repair and bundling (manylinux compliance)
- NumPy's built-in test suite (`np.test()`) - Linalg subset only
- SciPy's built-in test suite (`scipy.test()`) - Linalg subset validation
- pytest - MKL-service testing (via `pytest -s -v --pyargs mkl`)
- Nix/NixOS - Development environment (flake.nix, flake.lock)
- GitHub Actions - CI/CD orchestration
## Key Dependencies
- `packaging>=24.2` - Version handling and specifier parsing in build tools
- `pyyaml>=6.0.3` - YAML config parsing (ci-targets.yaml)
- `requests>=2.32.3` - HTTP requests for GitHub API (release asset fetching, version detection)
- `tabulate>=0.9.0` - Formatted output for build matrix/status tables
- MKL (Intel Math Kernel Library) - Compiled at build time, bundled into wheels
- numpy/numpy (IntelPython fork) - Source repo for NumPy with ILP64 patches
- IntelPython/mkl-service - MKL runtime service repo
- Intel MKL runtime libraries - Runtime dependency for BLAS/LAPACK operations
- `mkl-devel` - Build-time MKL development files (headers, pkg-config)
## Configuration
- Configured via GitHub Actions secrets (GITHUB_TOKEN for API calls)
- Build matrix configured in `ci-targets.yaml` (Python versions, runners, platforms)
- LD_LIBRARY_PATH - Set at build and test time for MKL library discovery
- `pyproject.toml` - Project metadata, dependencies, and tool configuration (ruff, uv)
- No setup.py or setup.cfg (modern pyproject.toml-only approach)
- Build args applied via `-Csetup-args=*` flags:
## Code Quality
- ruff 0.15+ - Fast Python linter and formatter
- Configuration:
## Platform Requirements
- uv ~0.9 required (enforced via `pyproject.toml` `required-version`)
- Python >=3.11 (via `requires-python`)
- Nix (optional, for reproducible dev environment)
- GitHub Actions runners:
- Python setup via `astral-sh/setup-uv@v7.1.6` action
- manylinux_2_28 compliance (Linux)
- macOS 10.9+ (universal2 for Intel/ARM)
- Windows 10+ (x86_64, arm64)
- MKL libraries bundled into wheels via auditwheel (no separate MKL install required)
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- Lowercase with underscores for executable scripts: `fetch_matrix`, `fetch_stats`, `store_info.py`
- Python modules: lowercase with underscores: `numpy_tests.py`, `scipy_tests.py`, `benchmarks.py`
- Directories: lowercase with hyphens (e.g., `.github/workflows`)
- Lowercase with underscores: `get_package_version()`, `parse_release_tag()`, `fetch_dict()`
- Verb-first naming for actions: `fetch_*`, `get_*`, `parse_*`, `run_*`, `setup()`
- Getter methods use `fetch_` or `get_` prefix
- Lowercase with underscores: `package_name`, `requires_python`, `python_versions`, `force_build`
- Loop variables and temporary variables: short names like `r`, `n`, `t` (acceptable in local context)
- Constants: `UPPERCASE_WITH_UNDERSCORES`: `REQUESTS_TIMEOUT = 45`, `PYTHON_VERSIONS`, `PYPI` (URLs)
- PascalCase: `FetchPackageData`, `FetchBuildMatrix`, `Config`, `Benchmark`, `Build`
- Exception classes: `FetchPackageDataError`
- Private attributes use leading underscore: `headers_` (used to avoid name conflicts with functions)
- Instance attributes stored as class properties
## Code Style
- Tool: `ruff` (pyproject.toml configured)
- Line length: 88 characters
- Indent width: 4 spaces
- Quote style: single quotes (`'string'` not `"string"`)
- Line endings: LF
- Tool: `ruff` with comprehensive rule set
- Key rules enabled (from pyproject.toml):
## Import Organization
- No path aliases used; imports are direct relative or absolute
- `ruff` handles import sorting (rule `I` enabled)
- Comments after violation: `# noqa: PLR2004` for literal comparisons
## Error Handling
- Custom exceptions inherit from `Exception`: `class FetchPackageDataError(Exception): pass`
- Broad exception catching when appropriate: `except Exception as e` in test runners
- Specific exceptions where possible: `except InvalidVersion`
- Raise with descriptive messages: `raise FetchPackageDataError(f'Download failed with status code {rc}')`
## Logging
- Direct print statements for test output and progress: `print(f"Test: {name}...")`
- Status reporting in tests: `print("  PASSED")`, `print(f"  FAILED: {type(e).__name__}: {e}")`
- Formatted output for results: `fmt.format(name=name, time=best, loops=loops)`
- No logging module used; output is CLI-oriented
## Comments
- Comment algorithm decisions or non-obvious logic
- Document assumptions or preconditions (seen in `numpy_tests.py` line 25-27)
- Explain why, not what (code should be clear about what it does)
- Not used; module-level docstrings present
- Module docstrings explain purpose and validation goals: `numpy_tests.py` lines 1-8
- Function docstrings used: `run_test(name, fn)` at line 53
## Function Design
- Keyword arguments used for optional parameters: `github=False, token=None`
- Type hints not used consistently (Python 3.11+ but project doesn't enforce them)
- Boolean flags use `action='store_true'` in argparse
- Return dictionaries with descriptive keys: `{'name': ..., 'version': ..., 'python': ..., 'os': ..., 'mkl': ...}`
- Return strings or None when appropriate
- Walrus operator (`:=`) used for assignment in conditionals: `if (rc := response.status_code) != 200:`
## Module Design
- No `__all__` defined; modules use implicit public interface
- Classes and functions are public; modules are tools meant to be run directly
- Not used; single-module organization
- `if __name__ == '__main__':` pattern used for CLI tools
- argparse for command-line argument parsing in most tools
## Modern Python Features
- Used to assign and check in one expression: `if (rc := response.status_code) != 200:`
- Used in regex matching: `if m := re.match(pattern, url):`
- Makes code more concise where appropriate
- Used for all string formatting: `f'Download failed with status code {rc}'`
- Formatted fields: `f'{value:.2e}'`, `f'{value:.6g}'`
- Not enforced in codebase; minimal usage
- Some functions accept Any types implicitly
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- Multi-platform cross-compilation (Linux, macOS, Windows)
- Patch-based modification system for upstream NumPy/SciPy/mkl-service
- Build metadata tracking (build.json) for incremental/conditional builds
- ILP64 (64-bit integer) BLAS/LAPACK configuration to overcome LP64 32-bit limits
- Python tooling-based orchestration using uv, requests, yaml, and packaging utilities
## Layers
- Purpose: Store and manage build information, target definitions, and version pinning
- Location: `build.json`, `ci-targets.yaml`, `pyproject.toml`, `flake.nix`
- Contains: Python version targets, platform definitions, MKL versions, build state
- Depends on: None (foundation layer)
- Used by: CI/CD workflows, build orchestration scripts
- Purpose: Fetch upstream packages, apply patches, manage matrix builds
- Location: `.github/workflows/` (wheels.yml, build_wheels.yml, build_manylinux_wheels.yml)
- Contains: GitHub Actions workflows that call build tools
- Depends on: Tools, metadata layer
- Used by: GitHub Actions CI system
- Purpose: Implement build matrix generation, patching, versioning, and index management
- Location: `tools/` directory
- Contains: Python scripts (fetch_matrix2, store_info.py, make_index2, etc.) and shell scripts
- Depends on: requests, packaging, pyyaml, tabulate libraries; external PyPI/GitHub APIs
- Used by: Workflows, release processes, index generation
- Purpose: Apply targeted modifications to upstream code for ILP64 configuration
- Location: `patches/numpy/`, `patches/scipy/`, `patches/mkl-service/`, `patches/mkl/`
- Contains: git patch files for each package
- Depends on: Upstream source code (checked out separately)
- Used by: Build orchestration (applied in CI before compilation)
- Purpose: Verify ILP64 configuration and numerical correctness
- Location: `tools/numpy_tests.py`, `tools/scipy_tests.py`
- Contains: Configuration assertions and numerical tests
- Depends on: Built numpy/scipy wheels
- Used by: CI/CD for validation (post-build)
- Purpose: Measure performance of BLAS/LAPACK operations
- Location: `benchmarks/benchmarks.py`
- Contains: Timing tests for linear algebra operations (eig, svd, solve, qr, etc.)
- Depends on: numpy (and optionally scipy)
- Used by: Manual performance analysis, not automated in CI
- Purpose: Provide Nix flake definitions for local development and deployment
- Location: `nix/`, `flake.nix`, `templates/nix/`
- Contains: Nix derivations for numpy, scipy, mkl-service, and MKL dependencies
- Depends on: nixpkgs, upstream package definitions
- Used by: Nix-based development and packaging workflows
## Data Flow
- **build.json**: Persistent store of all completed builds with MKL versions
- **Matrix generation**: Determines which builds to execute
## Key Abstractions
- Purpose: Represents one wheel (e.g., numpy-2.2.0-cp313-linux_x86_64)
- Examples: `Build` class in `tools/store_info.py`
- Pattern: Data class holding package name, version, python version, os, mkl version
- Purpose: Apply source code modifications for ILP64 configuration
- Examples: `patches/numpy/init_mkl.patch` (adds mkl import to _distributor_init.py)
- Pattern: Plain git patch files applied with `git apply` in CI
- Purpose: Standardized timing test for BLAS/LAPACK operations
- Examples: `Eig`, `SVD`, `Solve` classes in `benchmarks/benchmarks.py`
- Pattern: Subclass of Benchmark base class with setup() and time_it() methods
- Purpose: Confirm MKL ILP64 is active and numerical results are correct
- Examples: Tests in `tools/numpy_tests.py` (test_eigendecomp, test_svd, test_solve)
- Pattern: Assert BLAS/LAPACK name contains 'ilp64', run numpy/scipy linalg tests with strict errors
## Entry Points
- Location: `.github/workflows/wheels.yml`
- Triggers: Manual dispatch, scheduled push to main
- Responsibilities: Orchestrate matrix generation, build across platforms, validate, upload
- Location: `tools/fetch_matrix2`
- Triggers: Called from workflows/wheels.yml meta job
- Responsibilities: Query upstream releases, intersect with ci-targets.yaml, skip already-built, output JSON matrix
- Location: `tools/numpy_tests.py`, `tools/scipy_tests.py`
- Triggers: Called from CI after wheel built and installed
- Responsibilities: Assert ILP64 active, run numerical tests, run numpy/scipy linalg test suites
- Location: `benchmarks/run_benchmarks` (shell wrapper) or `python benchmarks/benchmarks.py`
- Triggers: Manual invocation
- Responsibilities: Time BLAS/LAPACK operations, output results to file or stdout
## Error Handling
- **Configuration validation** (`tools/numpy_tests.py` lines 29-42): Assert BLAS/LAPACK names contain 'ilp64' before numerical tests. Prevents silent LP64 mismatch bugs.
- **Reconstruction error checks** (`tools/numpy_tests.py` line 90): Verify eigendecomposition reconstruction error < 1e-8. Large errors indicate LP64/ILP64 mismatch.
- **CI matrix bail-out** (`tools/fetch_matrix2`): If no builds to do (all cached in build.json), matrix outputs null and build job skips (`if: ${{ needs.meta.outputs.matrix != 'null' }}`)
- **Timeout protection** (`tools/numpy_tests.py`, `tools/scipy_tests.py`): Run numpy.test() and scipy.test() with `--timeout=120` to catch infinite loops
## Cross-Cutting Concerns
- pyproject.toml (numpy-mkl version 0.1.9)
- ci-targets.yaml (Python versions 3.11-3.14)
- MKL versions pinned per build in build.json
- Upstream packages pinned by release tag in workflows
- MKL libraries installed via uv in isolated venv during build
- mkl-service and mkl added to built wheels' pyproject.toml via `uv add --no-sync`
- Runtime MKL version pinned to match compile-time version
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
