# MatX Performance Benchmarks

The benchmark target is optional and does not add cases to CTest:

~~~sh
cmake -S . -B build-verify \
  -DCMAKE_BUILD_TYPE=Release \
  -DMATX_BUILD_TESTS=ON \
  -DMATX_ENABLE_GRAPHBLAS=ON \
  -DMATX_BUILD_BENCHMARKS=ON
cmake --build build-verify --target matx_benchmarks -j4
python3 benchmarks/check_api_coverage.py --build-dir build-verify
~~~

Run the default scale sweep with:

~~~sh
./build-verify/benchmarks/matx_benchmarks > build-verify/matx-bench.csv
~~~

The sweep measures real and complex vector BLAS operations from 32 elements
through 1,048,576; dense GEMM, GEADD, GEMV, Hadamard, transpose, norms,
triangular solves, rank-1 and rank-k updates, inverse, and real matrix
exponential; and dense LU, Cholesky, least-squares, QR, eigenvalue, SVD,
determinant, and condition-number paths. Sparse coverage includes LU and
Cholesky factorization and solves where supported, COO-to-CSC conversion,
SpMV/SpMM, COO transforms, addition, reductions, extraction, dense conversion,
and scaling. Product sweeps cover dimensions 64 through 8192 at densities
0.01%, 0.1%, and 1%. Dense streaming operations include row-major and
column-major storage; SpMM uses 16 dense right-hand sides. Use `--quick` for a
short run or `--full` to include larger cases.

Use `./build-verify/benchmarks/matx_benchmarks --coo-conversion-only` to isolate
MatX's COO-to-CSR and COO-to-CSC conversions. It covers small, medium, 100,000-
and 131,072-order matrices, shuffled and column/row-sorted CSC inputs, and real
and complex CSC values. Each row checks sorted indices and conversion output;
CSC rows also validate the COO-to-CSC index map. The run reports estimated
temporary scratch bytes: unsorted CSC conversion uses an 8-byte source-index
array when any column has at least 256 entries, otherwise it uses 16-byte
`(row, input-index)` records; sorted CSC takes a no-scratch path. CSR reuses its
row-pointer array as scatter cursors without a separate offset array.

Core rows measure vector, dense, and sparse allocation/free plus vector and
dense fill. I/O rows measure dense and COO sparse Matrix Market read/write for
real and complex values. These cases verify dimensions, coordinates, and
round-tripped values; temporary files are removed after each case. Sparse
utility rows include nnz counts, absolute row/column sums, diagonal and
single-row/column extraction. Dense and sparse solve rows validate residuals;
inverse, QR, eigen, and SVD cases validate representative result properties.

Output is CSV, including estimated effective memory bandwidth for vector,
dense, and sparse kernels. Paired rows such as `vec_scal_pair` and
`vec_axpy_pair` time two inverse updates per call to keep the data bounded.
setup_us measures MatX object creation and input initialization;
first_us includes first-call setup such as backend conversion; steady_us is
the median of seven samples, each calibrated to about 10 ms. SpMV/SpMM outputs
are checked against a CPU reference; solve rows check residuals; and GEMM,
inverse, QR, eigen, and SVD rows check representative results.
The run exits with an error if a measured result fails its numerical check.
Backend and compiler details are written to stderr.

## Sparse solver comparison

The Ginkgo, PETSc, and hypre iterative backends are compared with MatX's direct
SuiteSparse KLU, CHOLMOD, UMFPACK, CXSparse, SuperLU, and MUMPS paths. Disabled
or unsupported direct backends are emitted as `skipped`. The comparison
executable is separate from the GraphBLAS benchmark target. CMake builds each
enabled iterative backend from its pinned upstream source, installs it below
the build tree, then builds and runs the comparison:

~~~sh
cmake -S . -B build-iterative \
  -DCMAKE_BUILD_TYPE=Release \
  -DMATX_BUILD_TESTS=ON \
  -DMATX_ENABLE_GINKGO=ON \
  -DMATX_ENABLE_PETSC=ON \
  -DMATX_ENABLE_HYPRE=ON \
  -DMATX_BUILD_ITERATIVE_BENCHMARKS=ON
cmake --build build-iterative --target matx_sparse_iterative_benchmarks matx_tests -j4
MATX_TEST_FILTER=sparse_iterative ./build-iterative/tests/matx_tests
for n in 1024 4096 10000 25000 50000 100000; do
  OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores \
    ./build-iterative/matx_sparse_iterative_benchmarks --size "$n" --repeats 3 \
    > "build-iterative/sparse-solver-comparison-$n.csv"
done
~~~

The comparison uses a 2D Poisson SPD matrix with CG and a nonsymmetric
convection-diffusion matrix with GMRES. `--size` selects an exact unknown count
on a near-square 2D grid; the listed sweep ranges from 1,024 to 100,000
unknowns. The legacy `--side` option still selects a square grid by width.
Each solver prepares the fixed matrix once, then solves a warm-up RHS followed
by distinct RHS vectors generated from known solutions. The CSV reports
`setup_ms` separately from `median_solve_ms`; compare solvers using the latter.
It also records iteration count, true residual, relative residual, relative
solution error, and status. Direct `setup_ms` measures factorization, including
the backend's symbolic/numeric work; COO matrix creation occurs before timing.
Timed RHS generation is outside the solve timer. Timed direct samples reuse
that factor. Iterative `setup_ms` covers solver creation, matrix import, and
the explicit preconditioner setup call; backend work deferred until the first
solve is handled by the discarded warm-up. Timed samples reuse that solver and
preconditioner. Direct rows cover LU and CHOLMOD
Cholesky for the SPD matrix. Iterative rows cover CG and GMRES with Jacobi and
backend-supported AMG/ILU. Iterative solve time includes MatX backend-vector
handling and true-residual checking. Unsupported combinations are emitted as
`skipped`; a setup, solve, or numerical-check failure makes the program exit
nonzero. Pin thread counts and keep CPU/compiler settings identical when
comparing CSV files. The example uses four OpenMP threads and three timed
samples per case; the default problem size is 65,536 unknowns.

`MATX_TEST_FILTER` accepts a substring of test names so backend tests can run
without unrelated tests. A filter that selects no tests exits with status 2.

The sparse API path uses direct COO accumulation up to 3,000,000 nonzeros for
SpMV, and up to both 3,000,000 nonzeros and 50,000,000 nonzero/RHS products
for SpMM; larger operations use GraphBLAS. `spmv_coo` and `spmm_coo` rows always
time the direct COO reference path. Below the limits the API rows use that same
direct path; above them, the rows measure GraphBLAS. The limits were raised
after the full benchmark showed direct COO accumulation faster at 671,088 and
2,684,354 nonzeros for SpMV and SpMM, including 16 RHS columns. Recheck the
crossover on the target CPU before tuning it.

See [OPTIMIZATION_PLAN.md](OPTIMIZATION_PLAN.md) for the staged measurement and
optimization plan. The benchmark uses representative workload families rather
than every scalar type and parameter combination supported by every API.
`check_api_coverage.py` compares MatX symbols exported by the built static
libraries with call sites in the benchmark source. It excludes internal
backend factory symbols that have no public header declarations. This verifies
that every public entry point has a benchmark call site; it does not prove
every API argument, dimension, storage layout, or backend combination was
exercised. Runtime cases also validate representative numerical results,
residuals, or round trips.

For comparisons, use the same machine, compiler, build type, and thread settings.
Set BLAS and OpenMP thread counts explicitly (for example,
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4) and record the CPU model, memory, and
command alongside the CSV.
