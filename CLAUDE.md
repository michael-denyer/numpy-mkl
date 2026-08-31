<!-- GSD:project-start source:PROJECT.md -->
## Project

**numpy-mkl: Windows Wheel Support**

A CI/CD pipeline that builds ILP64 NumPy and mkl-service wheels linked against Intel MKL, published to a GitHub Pages PyPI index. Currently Linux-only. This milestone adds Windows wheel builds.

**Core Value:** Ship ILP64 NumPy wheels for Linux and Windows that are reliable — correct MKL linkage, passing tests, bundled shared libraries.

### Constraints

- **Platform**: Windows builds use `windows-latest` GitHub runner, no container
- **Compiler**: MSVC via Visual Studio (meson `--vsenv` flag)
- **DLL bundling**: Must use delvewheel (auditwheel is Linux-only)
- **pkg-config**: Not available by default on Windows, needs choco install
- **Compatibility**: Changes must not break existing Linux builds
<!-- GSD:project-end -->
