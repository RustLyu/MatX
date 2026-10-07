# MatX Performance Optimization Plan

Optimize every public numerical module while preserving each API's numerical
results and backend behavior. Benchmark first, then change one implementation
path at a time and compare it with the recorded baseline.

## Coverage

| Module | Operation families | Workload dimensions |
| --- | --- | --- |
| `vec_blas` | scale, copy, swap, dot, norms, axpy, cross | lengths from tiny vectors through cache and memory sized vectors; real and complex |
| `dense_blas` | GEMM/GEMV, matrix updates, triangular operations, rank-k updates, transpose, elementwise operations, norms, inverse, exponential | square and rectangular matrices; small, cache sized, and large cases; row/column major where supported; real and complex |
| `sparse_blas` | SpMV/SpMM, sparse products, conversion, transpose, add, reductions, norms, diagonal and scaling operations | dimensions, density, row distribution, RHS count, and real/complex values |
| `dense_solve` | LU, Cholesky, least squares, QR, eigenproblems, determinant, condition number, SVD | factor-only, reused-factor solve, and one-shot paths over increasing orders; real/complex where supported |
| `sparse_solve` | LU/Cholesky, reused-factor solve, one-shot solve, COO-to-CSC conversion | order, nonzeros, density, matrix structure, and available solver backends |
| `core` | vector, dense, and sparse allocation and access | allocation size, contiguous/strided access, and repeated reuse |
| `io` | Matrix Market read/write | file size, density, real/complex values, and parsing versus allocation cost |

## Measurement and Optimization Order

1. Record the current backend, compiler, CPU, thread count, build flags, and baseline timings. Separate setup/conversion, first-call, steady-state, and allocation costs.
2. Cover small inputs where dispatch and allocation dominate, cache sized inputs, and large inputs where bandwidth or compute throughput dominates. Validate outputs against references or residuals for every benchmark case.
3. Rank opportunities by end-to-end time and frequency. Optimize high-impact vector and dense paths, then sparse kernels and conversions, then solver setup/reuse and container overhead. Keep reference paths for correctness comparisons.
4. Re-run affected module tests and the full suite after each change. Retain an optimization only when results are correct and repeated measurements show a benefit without a material regression at other sizes.

The executable records representative baselines for each module family:
vector BLAS; dense BLAS streaming, triangular solves, rank updates, inversion,
and exponential; sparse products, COO utilities, and COO-to-CSC conversion;
dense LU, Cholesky, least-squares, QR, eigen, SVD, determinant, and condition
number; sparse LU and Cholesky; core allocation and fill; and dense/sparse
Matrix Market I/O. It validates measured outputs with CPU references, residuals,
round trips, or result properties as appropriate. It does not enumerate every
scalar type, layout, or backend combination for every entry point.
Run `python3 benchmarks/check_api_coverage.py --build-dir build-verify` after
building the benchmark to confirm each public symbol exported by the built
libraries has a call site in the benchmark source. The check excludes internal
backend factories that are not declared in public headers.
