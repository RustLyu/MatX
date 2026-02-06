## MatX

`MatX` is a C-based linear algebra library split into 4 subprojects:

- **core**: basic data types (dense matrices, allocators, error codes)
- **compute**: numerical compute on core types (BLAS backend: reference / OpenBLAS / BLIS)
- **solve**: linear solves on core types (temporarily via SuiteSparse; extensible backend interface)
- **tests**: unit tests using GoogleTest

### Build (Windows / Linux)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMATX_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release
```

### Notes

- If OpenBLAS/BLIS/SuiteSparse are not found, MatX still builds:
  - `compute` uses a reference DGEMM implementation
  - `solve` returns `MATX_ERR_NOT_SUPPORTED` (SuiteSparse integration to be added)

