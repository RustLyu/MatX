# MatX Optimization Status

This file records the active performance work so a later session can continue
without repeating completed measurements.

## User Priorities

- Optimize MatX's own numerical and linear-solver code, including COO/CSR/CSC
  conversions and memory use. For LU solve work, focus on forward/back
  substitution time rather than LU factorization.
- Use Linux perf counters to profile the full interface benchmark and make
  targeted changes. Do not limit perf work to sparse format conversions.
- The previously requested 24-hour run was explicitly stopped. Do not restart a
  long-running test without a new request; finite benchmark runs are active.
- Continue authorized work to completion. Present major architectural changes
  for the user's decision before starting them.
- The workspace already contains unrelated dirty and untracked files. Preserve
  them; do not reset or clean the repository.

## Completed: Adaptive COO-to-CSC Sorting

In `sparse_solve/src/tools.c`, unsorted COO-to-CSC now selects a path based on
column occupancy:

- If every column has fewer than 256 entries, use the existing 16-byte
  `(row, source-index)` records and introsort. This preserves the low-occupancy
  algorithm and avoids indirect index sorting.
- If any column has at least 256 entries, scatter 8-byte source indices, radix
  sort large columns by 64-bit row, and sort smaller buckets in a 256-entry
  stack buffer. The existing `coo2csc` output storage is reused as radix scratch
  before mappings are emitted.
- Duplicate-row merging, strictly sorted CSC rows, and COO-to-CSC mappings are
  preserved. The Entry allocation has a `SIZE_MAX` overflow check.
- A regression test exercises the radix path with 64-bit row indices near
  `INT64_MAX` and checks duplicate merging and every input mapping.

The COO conversion benchmark estimates the selected scratch size and validates
column pointers, sorted rows, and all mappings. The README documents the 256
entry threshold.

## Validation and Measurements

- Release build: `cmake --build build-verify --target matx_benchmarks matx_tests -j2`
  passed.
- `ctest --test-dir build-verify --output-on-failure` passed; direct test binary
  reported 197 tests, 2205 assertions, 0 failures.
- `python3 benchmarks/check_api_coverage.py --build-dir build-verify` reported
  277/277 public exported symbols with benchmark call sites (10 internal
  factories excluded).
- Full interface benchmark completed with
  `OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4` and `--full --perf-counters`:
  1091 measured rows. Perf counters are scoped to the calling thread. The
  benchmark is representative coverage, not every backend/layout/argument
  combination.
- `git diff --check` passed.
- The high-occupancy 2048-order / 750,000-nnz COO-to-CSC case reduced temporary
  scratch from 12 MB to 6 MB. The three-run current median versus the earlier
  single old-version sample was roughly:

  | Type | Old time | Current median | Old cycles | Current median |
  | --- | ---: | ---: | ---: | ---: |
  | f64 | 32.33 ms | 28.23 ms | 116.1 M | 66.7 M |
  | c64 | 31.28 ms | 26.76 ms | 112.6 M | 65.4 M |

- Low-occupancy cases were generally close to baseline, with small differences
  that are not statistically conclusive because the old baseline was one run:
  f64 100k order was about 17.62 ms vs 17.35 ms old; c64 was 17.65 ms vs
  19.24 ms old. Current three-run CSC samples are under `/tmp/matx-perf-csc-isolated-*.csv`;
  the old sample is `/tmp/matx-perf-csc-introsort-ab.csv`.
- Full fixed-thread perf CSV:
  `/tmp/matx-perf-full-post-csc-4t.csv`.

## Next Candidate: COO SpMM With Multiple RHS

The fixed four-thread full perf run measured COO SpMM at order 16,384, 2,684,354
nonzeros, and 16 RHS at about 245 ms for f64 and 539 ms for c64. At this workload
the dispatch thresholds select MatX's direct COO kernel, even though the CSV
backend label says `GRAPHBLAS+COO`. The current kernel loops over nonzeros and
then RHS columns; column-major RHS/output accesses jump by a full matrix stride.
Packing or tiling the RHS dimension could improve locality and SIMD use, but
requires careful handling of layouts, strides, aliases, and temporary buffers.

Discuss this larger kernel redesign with the user before implementing it. The
reused dense solve also remains a candidate: at order 2048, f64 was about 4.55
ms and c64 about 10.34 ms under the four-thread full sweep. Keep factorization
time out of solve-only comparisons when returning to that work.
