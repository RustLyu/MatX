# Repository Guidelines

## Project Structure & Module Organization

MatX is a modular C17/C++20 library built with CMake. Public headers and implementation files live in each module's `include/matx/` and `src/` directories. `types/` and `core/` define types and containers; `*_blas/` modules provide compute wrappers; `*_solve/` modules provide solvers; `io/` handles Matrix Market files; `tools/` contains logging and timing utilities. Dependency logic is in `cmake/`, and tests are in `tests/test_*.cpp`.

## Build, Test, and Development Commands

- `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMATX_BUILD_TESTS=ON`: configure a Release build with tests.
- `cmake --build build -j$(nproc)`: build all MatX targets using the local generator.
- `ctest --test-dir build --output-on-failure`: run the registered test suite.
- `./build/tests/matx_tests`: run the GoogleTest binary directly when it is built.
- `ctest --test-dir build -R <filter> --output-on-failure`: run a subset of tests.

Dependencies are resolved through CMake `FetchContent` or CMake package discovery; avoid vendoring generated third-party source into the tree.

## Coding Style & Naming Conventions

Use existing C style in nearby files: tabs are common in C sources, two-space indentation appears in C++ tests and CMake. Public APIs use the `matx_` prefix and type suffixes such as `_f64` and `_c64`. Public headers belong under `*/include/matx/`; internal definitions should stay in internal headers or `src/`. Keep exported handle types opaque and preserve the vtable backend-dispatch pattern used by compute modules.

## Testing Guidelines

Tests use GoogleTest. Add new coverage in `tests/test_<area>.cpp` and use descriptive `TEST(suite, case)` names such as `TEST(core_dense, create_destroy)`. Cover both real and complex variants when changing numeric APIs, and include backend-not-supported expectations where optional backends may differ. Run `ctest --test-dir build --output-on-failure` before submitting.

## Commit & Pull Request Guidelines

Recent commit messages are short, imperative summaries such as `fix compile error` and `update README.md`; keep the first line concise and specific. Pull requests should describe the affected module, list build/test commands run, note backend or platform assumptions, and link related issues. Include screenshots only for user-visible documentation or tooling changes.

## Security & Configuration Tips

Do not commit local build directories, dependency trees, or machine-specific caches. Build GraphBLAS through SuiteSparse and keep dependencies, caches, and generated files inside the repository build directory; never use `~/.SuiteSparse`. For GraphBLAS failures, report compiler diagnostics before adding cache or JIT workarounds. Reproducible builds should state flags such as `-DMATX_BACKEND=OPENBLAS -DMATX_ENABLE_GRAPHBLAS=ON`.
