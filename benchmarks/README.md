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
