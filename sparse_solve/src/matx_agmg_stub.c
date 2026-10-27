/**
 * @file matx_agmg_stub.c
 * @brief AGMG (Algebraic Multigrid) backend for sparse linear systems.
 *
 * AGMG is an aggregation-based algebraic multigrid solver (iterative).
 * It is particularly fast for circuit simulation matrices (MNA).
 *
 * Reference: Y. Notay, "An aggregation-based algebraic multigrid method",
 *            Electronic Transactions on Numerical Analysis, 2010.
 *
 * Dependencies:
 *   - AGMG Fortran library + C wrapper (dagmg.h / zagmg.h)
 *   - Obtain from https://agmg.eu (free for academic use)
 *
 * AGMG C API (one-shot solve):
 *   dagmg_(&n, a, ja, ia, f, x, &ijob, &iprint, &nrest, &iter, &tol);
 *   zagmg_(&n, a, ja, ia, f, x, &ijob, &iprint, &nrest, &iter, &tol);
 *
 *   Parameters (AGMG 3.x, Fortran convention — CSR is 1-based):
 *     n      [in]  matrix dimension
 *     a      [in]  CSR values (double*)
 *     ja     [in]  CSR column indices (int*, 1-based)
 *     ia     [in]  CSR row pointers (int*, 1-based)
 *     f      [in]  right-hand side (double*)
 *     x      [out] solution vector (double*)
 *     ijob   [in]  control: 0=setup+solve+release (one-shot), 1=setup, 2=solve
 *     iprint [in]  verbosity (0=silent)
 *     nrest  [in]  GCR restart (1=Flexible CG for symmetric, >1=GCR)
 *     iter   [in/out] INPUT = max iterations, OUTPUT = iterations performed
 *     tol    [in]  relative tolerance
 *
 * Note: AGMG is an ITERATIVE solver. Unlike KLU/UMFPACK which are direct
 * solvers, AGMG approximately solves the system. Convergence is not
 * guaranteed for all matrix types. For well-conditioned or mildly
 * indefinite circuit MNA matrices, AGMG typically converges in < 20
 * V-cycles to 1e-12 tolerance.
 */

#include "matx/matx_sparse_solve.h"
#include "matx/matx_types_internal.h"
#include "matx/matx_log.h"

#include <limits.h>
#include <string.h>
#include <stdlib.h>

#if MATX_HAVE_AGMG
#include "dagmg.h"
#include "zagmg.h"
#endif

/* ── Internal: COO → CSR conversion (int-indexed, for AGMG) ──────────── */

typedef struct agmg_csr_d
{
    int n;
    int nnz;
    int* ia;      /* row pointers, length n+1 */
    int* ja;      /* column indices, length nnz */
    double* a;    /* values, length nnz */
} agmg_csr_d_t;

typedef struct agmg_csr_z
{
    int n;
    int nnz;
    int* ia;      /* row pointers, length n+1 */
    int* ja;      /* column indices, length nnz */
    double* a;    /* values: interleaved real/imag, length 2*nnz */
} agmg_csr_z_t;

/* ── Introsort for CSR row entries ────────────────────────────────────────
 *
 * Sorts column indices (ja) within a CSR row while keeping associated values
 * (a) in sync.  Uses introsort — quicksort with median-of-three pivot and
 * heap-sort fallback — for rows longer than 32; insertion sort for shorter rows.
 *
 * Two overloads: one for real (single double per entry), one for complex
 * (two interleaved doubles per entry).
 */

#define AGMG_SORT_INSERTION_LIMIT 32

static void swap_real(int* ja, double* a, int i, int j)
{
    int    jtmp = ja[i]; ja[i] = ja[j]; ja[j] = jtmp;
    double atmp = a[i];  a[i]  = a[j];  a[j]  = atmp;
}

static void sift_real_down(int* ja, double* a, int root, int end)
{
    for (;;) {
        int child = root * 2 + 1;
        if (child > end) return;
        if (child + 1 <= end && ja[child] < ja[child + 1]) ++child;
        if (ja[root] >= ja[child]) return;
        swap_real(ja, a, root, child);
        root = child;
    }
}

static void heapsort_real(int* ja, double* a, int len)
{
    for (int start = len / 2; start > 0;) { --start; sift_real_down(ja, a, start, len - 1); }
    for (int end = len - 1; end > 0; --end) {
        swap_real(ja, a, 0, end);
        sift_real_down(ja, a, 0, end - 1);
    }
}

static int median3_real(int* ja, int a, int b, int c)
{
    if (ja[a] < ja[b]) return (ja[b] < ja[c]) ? b : (ja[a] < ja[c] ? c : a);
    return (ja[a] < ja[c]) ? a : (ja[b] < ja[c] ? c : b);
}

static void introsort_real(int* ja, double* a, int len, unsigned depth_limit)
{
    /* quicksort phase */
    while (len > AGMG_SORT_INSERTION_LIMIT) {
        if (depth_limit == 0) { heapsort_real(ja, a, len); return; }
        --depth_limit;

        int pivot = median3_real(ja, 0, len / 2, len - 1);
        int jval  = ja[pivot];
        int left = 0, right = len - 1;
        for (;;) {
            while (ja[left]  < jval) ++left;
            while (ja[right] > jval) --right;
            if (left >= right) break;
            swap_real(ja, a, left, right);
            ++left; --right;
        }
        /* recurse on smaller partition, iterate on larger */
        int left_len  = right + 1;
        int right_len = len - left;
        if (left_len < right_len) {
            if (left_len > 1) introsort_real(ja, a, left_len, depth_limit);
            ja += left; a += left; len = right_len;
        } else {
            if (right_len > 1) introsort_real(ja + left, a + left, right_len, depth_limit);
            len = left_len;
        }
    }
    /* insertion sort for short subarrays */
    for (int i = 1; i < len; ++i) {
        int    jv = ja[i];
        double av = a[i];
        int j = i - 1;
        while (j >= 0 && ja[j] > jv) { ja[j + 1] = ja[j]; a[j + 1] = a[j]; --j; }
        ja[j + 1] = jv; a[j + 1] = av;
    }
}

/* ── complex (interleaved) variant ── */

static void swap_z(int* ja, double* a, int i, int j)
{
    int    jtmp  = ja[i];      ja[i]      = ja[j];      ja[j]      = jtmp;
    double artmp = a[2*i];     a[2*i]     = a[2*j];     a[2*j]     = artmp;
    double aitmp = a[2*i + 1]; a[2*i + 1] = a[2*j + 1]; a[2*j + 1] = aitmp;
}

static void sift_z_down(int* ja, double* a, int root, int end)
{
    for (;;) {
        int child = root * 2 + 1;
        if (child > end) return;
        if (child + 1 <= end && ja[child] < ja[child + 1]) ++child;
        if (ja[root] >= ja[child]) return;
        swap_z(ja, a, root, child);
        root = child;
    }
}

static void heapsort_z(int* ja, double* a, int len)
{
    for (int start = len / 2; start > 0;) { --start; sift_z_down(ja, a, start, len - 1); }
    for (int end = len - 1; end > 0; --end) {
        swap_z(ja, a, 0, end);
        sift_z_down(ja, a, 0, end - 1);
    }
}

static int median3_z(int* ja, int a, int b, int c)
{
    if (ja[a] < ja[b]) return (ja[b] < ja[c]) ? b : (ja[a] < ja[c] ? c : a);
    return (ja[a] < ja[c]) ? a : (ja[b] < ja[c] ? c : b);
}

static void introsort_z(int* ja, double* a, int len, unsigned depth_limit)
{
    while (len > AGMG_SORT_INSERTION_LIMIT) {
        if (depth_limit == 0) { heapsort_z(ja, a, len); return; }
        --depth_limit;

        int pivot = median3_z(ja, 0, len / 2, len - 1);
        int jval  = ja[pivot];
        int left = 0, right = len - 1;
        for (;;) {
            while (ja[left]  < jval) ++left;
            while (ja[right] > jval) --right;
            if (left >= right) break;
            swap_z(ja, a, left, right);
            ++left; --right;
        }
        int left_len  = right + 1;
        int right_len = len - left;
        if (left_len < right_len) {
            if (left_len > 1) introsort_z(ja, a, left_len, depth_limit);
            ja += left; a += 2 * left; len = right_len;
        } else {
            if (right_len > 1) introsort_z(ja + left, a + 2 * left, right_len, depth_limit);
            len = left_len;
        }
    }
    for (int i = 1; i < len; ++i) {
        int    jv  = ja[i];
        double avr = a[2*i], avi = a[2*i + 1];
        int j = i - 1;
        while (j >= 0 && ja[j] > jv) {
            ja[j + 1] = ja[j]; a[2*(j+1)] = a[2*j]; a[2*(j+1)+1] = a[2*j+1]; --j;
        }
        ja[j + 1] = jv; a[2*(j+1)] = avr; a[2*(j+1)+1] = avi;
    }
}

/**
 * @brief Convert a COO real matrix to CSR (int-indexed) for AGMG.
 *
 * AGMG requires 1-based (Fortran) CSR with int indices. Duplicate COO entries
 * are summed.  Uses introsort for rows with >32 entries; insertion sort
 * otherwise.  All allocations go through matx_malloc so user-provided
 * allocators work correctly.
 */
#if MATX_HAVE_AGMG
static matx_status_t coo_to_csr_d(matx_coo_d_i8_t A, agmg_csr_d_t* csr)
{
    if (A->nrows != A->ncols) {
        MATX_ERROR("%s: AGMG requires square matrix", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows > (matx_int64_t) INT_MAX) {
        MATX_ERROR("%s: matrix dimension > INT_MAX not supported by AGMG", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }

    const int n = (int) A->nrows;
    const matx_int64_t nnz64 = A->nnz;

    memset(csr, 0, sizeof(*csr));
    csr->n = n;
    csr->nnz = (int) nnz64;   /* nnz ≤ INT_MAX by construction */

    csr->ia = (int*) matx_malloc(&A->alloc, ((size_t) n + 1) * sizeof(int));
    csr->ja = (int*) matx_malloc(&A->alloc, (size_t) csr->nnz * sizeof(int));
    csr->a  = (double*) matx_malloc(&A->alloc, (size_t) csr->nnz * sizeof(double));
    if (!csr->ia || !csr->ja || !csr->a) {
        matx_free(&A->alloc, csr->ia); matx_free(&A->alloc, csr->ja);
        matx_free(&A->alloc, csr->a);
        memset(csr, 0, sizeof(*csr));
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(csr->ia, 0, ((size_t) n + 1) * sizeof(int));
    memset(csr->a,  0, (size_t) csr->nnz * sizeof(double));

    /* Count entries per row */
    for (matx_int64_t k = 0; k < nnz64; ++k) {
        int r = (int) A->rows[k];
        csr->ia[r + 1]++;
    }

    /* Prefix sum → row pointers */
    for (int i = 1; i <= n; ++i)
        csr->ia[i] += csr->ia[i - 1];

    /* Scatter into CSR; duplicates summed in-init (memset above) */
    int* w = (int*) matx_malloc(&A->alloc, (size_t) n * sizeof(int));
    if (!w) {
        matx_free(&A->alloc, csr->ia); matx_free(&A->alloc, csr->ja);
        matx_free(&A->alloc, csr->a);
        memset(csr, 0, sizeof(*csr));
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(w, 0, (size_t) n * sizeof(int));
    for (matx_int64_t k = 0; k < nnz64; ++k) {
        int r = (int) A->rows[k];
        int c = (int) A->columns[k];
        int pos = csr->ia[r] + w[r];
        csr->ja[pos] = c + 1;            /* AGMG uses 1-based column indices */
        csr->a [pos] += A->values[k];   /* sum duplicates */
        w[r]++;
    }
    matx_free(&A->alloc, w);

    /* Sort column indices within each row (AGMG expects sorted CSR).
     * Use introsort for wide rows, insertion sort for narrow rows. */
    for (int r = 0; r < n; ++r) {
        int beg = csr->ia[r];
        int len = csr->ia[r + 1] - beg;
        if (len < 2) continue;
        if (len <= AGMG_SORT_INSERTION_LIMIT) {
            for (int i = 1; i < len; ++i) {
                double aval = csr->a[beg + i];
                int    jval = csr->ja[beg + i];
                int j = i - 1;
                while (j >= 0 && csr->ja[beg + j] > jval) {
                    csr->a [beg + j + 1] = csr->a [beg + j];
                    csr->ja[beg + j + 1] = csr->ja[beg + j];
                    --j;
                }
                csr->a [beg + j + 1] = aval;
                csr->ja[beg + j + 1] = jval;
            }
        } else {
            unsigned depth = 0;
            for (int d = len; d > 1; d >>= 1) depth += 2;
            introsort_real(csr->ja + beg, csr->a + beg, len, depth);
        }
    }

    /* AGMG uses 1-based (Fortran) row pointers: shift ia by +1 */
    for (int i = 0; i <= n; ++i)
        csr->ia[i] += 1;

    return MATX_OK;
}

/**
 * @brief Convert a COO complex matrix to CSR (int-indexed) for AGMG.
 *
 * Complex values stored as interleaved double[2], compatible with
 * matx_complex_d_t memory layout.
 */
static matx_status_t coo_to_csr_z(matx_coo_z_i8_t A, agmg_csr_z_t* csr)
{
    if (A->nrows != A->ncols) {
        MATX_ERROR("%s: AGMG requires square matrix", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows > (matx_int64_t) INT_MAX) {
        MATX_ERROR("%s: matrix dimension > INT_MAX not supported by AGMG", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }

    const int n = (int) A->nrows;
    const matx_int64_t nnz64 = A->nnz;

    memset(csr, 0, sizeof(*csr));
    csr->n = n;
    csr->nnz = (int) nnz64;

    csr->ia = (int*)    matx_malloc(&A->alloc, ((size_t) n + 1) * sizeof(int));
    csr->ja = (int*)    matx_malloc(&A->alloc, (size_t) csr->nnz * sizeof(int));
    csr->a  = (double*) matx_malloc(&A->alloc, (size_t) csr->nnz * 2 * sizeof(double));
    if (!csr->ia || !csr->ja || !csr->a) {
        matx_free(&A->alloc, csr->ia); matx_free(&A->alloc, csr->ja);
        matx_free(&A->alloc, csr->a);
        memset(csr, 0, sizeof(*csr));
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(csr->ia, 0, ((size_t) n + 1) * sizeof(int));
    memset(csr->a,  0, (size_t) csr->nnz * 2 * sizeof(double));

    for (matx_int64_t k = 0; k < nnz64; ++k) {
        int r = (int) A->rows[k];
        csr->ia[r + 1]++;
    }
    for (int i = 1; i <= n; ++i)
        csr->ia[i] += csr->ia[i - 1];

    int* w = (int*) matx_malloc(&A->alloc, (size_t) n * sizeof(int));
    if (!w) {
        matx_free(&A->alloc, csr->ia); matx_free(&A->alloc, csr->ja);
        matx_free(&A->alloc, csr->a);
        memset(csr, 0, sizeof(*csr));
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(w, 0, (size_t) n * sizeof(int));
    for (matx_int64_t k = 0; k < nnz64; ++k) {
        int r = (int) A->rows[k];
        int c = (int) A->columns[k];
        int pos = csr->ia[r] + w[r];
        csr->ja[pos] = c + 1;            /* AGMG uses 1-based column indices */
        csr->a [2 * pos]     += A->values[k].real;
        csr->a [2 * pos + 1] += A->values[k].imag;
        w[r]++;
    }
    matx_free(&A->alloc, w);

    for (int r = 0; r < n; ++r) {
        int beg = csr->ia[r];
        int len = csr->ia[r + 1] - beg;
        if (len < 2) continue;
        if (len <= AGMG_SORT_INSERTION_LIMIT) {
            for (int i = 1; i < len; ++i) {
                int    jval = csr->ja[beg + i];
                double ar   = csr->a [2 * (beg + i)];
                double ai   = csr->a [2 * (beg + i) + 1];
                int j = i - 1;
                while (j >= 0 && csr->ja[beg + j] > jval) {
                    csr->ja[beg + j + 1] = csr->ja[beg + j];
                    csr->a [2 * (beg + j + 1)]     = csr->a[2 * (beg + j)];
                    csr->a [2 * (beg + j + 1) + 1] = csr->a[2 * (beg + j) + 1];
                    --j;
                }
                csr->ja[beg + j + 1] = jval;
                csr->a [2 * (beg + j + 1)]     = ar;
                csr->a [2 * (beg + j + 1) + 1] = ai;
            }
        } else {
            unsigned depth = 0;
            for (int d = len; d > 1; d >>= 1) depth += 2;
            introsort_z(csr->ja + beg, csr->a + 2 * beg, len, depth);
        }
    }

    /* AGMG uses 1-based (Fortran) row pointers: shift ia by +1 */
    for (int i = 0; i <= n; ++i)
        csr->ia[i] += 1;

    return MATX_OK;
}

static void agmg_csr_d_free(const matx_alloc_t* alloc, agmg_csr_d_t* csr)
{
    matx_free(alloc, csr->ia); matx_free(alloc, csr->ja); matx_free(alloc, csr->a);
    memset(csr, 0, sizeof(*csr));
}

static void agmg_csr_z_free(const matx_alloc_t* alloc, agmg_csr_z_t* csr)
{
    matx_free(alloc, csr->ia); matx_free(alloc, csr->ja); matx_free(alloc, csr->a);
    memset(csr, 0, sizeof(*csr));
}

/* ── Factor / Solve / Destroy (real) ──────────────────────────────────── */

static matx_status_t agmg_factor_csc_d_i8(const matx_alloc_t* alloc,
                                           matx_coo_d_i8_t A,
                                           matx_factor_sparse_d_i8_t* out_F)
{
    if (!A || !out_F) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    /* Destroy previous factorization if present */
    if (out_F->reserved) {
        agmg_csr_d_t* old = (agmg_csr_d_t*) out_F->reserved;
        agmg_csr_d_free(alloc, old);
        matx_free(alloc, old);
        out_F->reserved = NULL;
    }

    agmg_csr_d_t* csr = (agmg_csr_d_t*) matx_malloc(alloc, sizeof(*csr));
    if (!csr) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    matx_status_t st = coo_to_csr_d(A, csr);
    if (st != MATX_OK) {
        matx_free(alloc, csr);
        return st;
    }

    out_F->reserved = csr;
    return MATX_OK;
}

static matx_status_t agmg_solve_csc_d_i8(const matx_alloc_t* alloc,
                                          matx_factor_sparse_d_i8_t* F,
                                          const matx_double* b,
                                          matx_double* x)
{
    if (!F || !F->reserved || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    agmg_csr_d_t* csr = (agmg_csr_d_t*) F->reserved;

    /* Copy RHS (AGMG may modify it in-place) */
    double* f = (double*) matx_malloc(alloc, (size_t) csr->n * sizeof(double));
    if (!f) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memcpy(f, b, (size_t) csr->n * sizeof(double));
    memset(x, 0, (size_t) csr->n * sizeof(double));

    int ijob   = 0;     /* 0 = one-shot solve (setup + solve + release) */
    int iprint = 0;     /* silent */
    int nrest  = 10;    /* GCR restart (default; use 1 for symmetric SPD) */
    int iter   = 100;   /* INPUT: maximum number of iterations */
    double tol = 1e-12;

    dagmg_(&csr->n, csr->a, csr->ja, csr->ia, f, x,
           &ijob, &iprint, &nrest, &iter, &tol);

    /* iter on output is the number of iterations actually performed.
     * AGMG 3.x has no reverse-communication interface (ijob is input-only),
     * so a non-converged solve is signalled by iter reaching the cap. */
    if (iter >= 100) {
        MATX_ERROR("%s: AGMG did not converge within %d iterations", __func__, iter);
        matx_free(alloc, f);
        return MATX_ERR_INTERNAL;
    }

    matx_free(alloc, f);
    return MATX_OK;
}

static void agmg_factor_csc_d_i8_destroy(const matx_alloc_t* alloc,
                                          matx_factor_sparse_d_i8_t* F)
{
    if (!F || !F->reserved) return;
    agmg_csr_d_t* csr = (agmg_csr_d_t*) F->reserved;
    agmg_csr_d_free(alloc, csr);
    matx_free(alloc, csr);
    F->reserved = NULL;
}

/* ── Factor / Solve / Destroy (complex) ───────────────────────────────── */

static matx_status_t agmg_factor_csc_z_i8(const matx_alloc_t* alloc,
                                           matx_coo_z_i8_t A,
                                           matx_factor_sparse_z_i8_t* out_F)
{
    if (!A || !out_F) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    if (out_F->reserved) {
        agmg_csr_z_t* old = (agmg_csr_z_t*) out_F->reserved;
        agmg_csr_z_free(alloc, old);
        matx_free(alloc, old);
        out_F->reserved = NULL;
    }

    agmg_csr_z_t* csr = (agmg_csr_z_t*) matx_malloc(alloc, sizeof(*csr));
    if (!csr) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    matx_status_t st = coo_to_csr_z(A, csr);
    if (st != MATX_OK) {
        matx_free(alloc, csr);
        return st;
    }

    out_F->reserved = csr;
    return MATX_OK;
}

static matx_status_t agmg_solve_csc_z_i8(const matx_alloc_t* alloc,
                                          matx_factor_sparse_z_i8_t* F,
                                          const matx_vec_z_i8_t b,
                                          matx_vec_z_i8_t x)
{
    if (!F || !F->reserved || !b || !b->data || !x || !x->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    agmg_csr_z_t* csr = (agmg_csr_z_t*) F->reserved;

    /* Copy RHS (AGMG may modify in-place) */
    const size_t bytes = (size_t) csr->n * sizeof(matx_complex_d_t);
    double* f = (double*) matx_malloc(alloc, bytes);
    if (!f) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memcpy(f, b->data, bytes);
    memset(x->data, 0, bytes);

    int ijob   = 0;
    int iprint = 0;
    int nrest  = 10;
    int iter   = 100;   /* INPUT: maximum number of iterations */
    double tol = 1e-12;

    zagmg_(&csr->n, csr->a, csr->ja, csr->ia, f, (double*) x->data,
           &ijob, &iprint, &nrest, &iter, &tol);

    if (iter >= 100) {
        MATX_ERROR("%s: AGMG did not converge within %d iterations", __func__, iter);
        matx_free(alloc, f);
        return MATX_ERR_INTERNAL;
    }

    matx_free(alloc, f);
    return MATX_OK;
}

static void agmg_factor_csc_z_i8_destroy(const matx_alloc_t* alloc,
                                          matx_factor_sparse_z_i8_t* F)
{
    if (!F || !F->reserved) return;
    agmg_csr_z_t* csr = (agmg_csr_z_t*) F->reserved;
    agmg_csr_z_free(alloc, csr);
    matx_free(alloc, csr);
    F->reserved = NULL;
}

/* ── Backend factory ──────────────────────────────────────────────────── */

matx_sparse_linsolve_t matx_linsolve_make_agmg(matx_alloc_t alloc)
{
    matx_sparse_linsolve_t ls;
    memset(&ls, 0, sizeof(ls));
    ls.kind  = MATX_LINSOLVE_BACKEND_AGMG;
    ls.alloc = alloc;

    ls.vt.factor_csc_d_i8          = agmg_factor_csc_d_i8;
    ls.vt.solve_csc_d_i8            = agmg_solve_csc_d_i8;
    ls.vt.factor_csc_d_i8_destroy   = agmg_factor_csc_d_i8_destroy;

    ls.vt.factor_csc_z_i8          = agmg_factor_csc_z_i8;
    ls.vt.solve_csc_z_i8            = agmg_solve_csc_z_i8;
    ls.vt.factor_csc_z_i8_destroy   = agmg_factor_csc_z_i8_destroy;

    /* AGMG does not provide Cholesky or refactorization */
    ls.vt.factor_chol_csc_d_i8        = NULL;
    ls.vt.solve_chol_csc_d_i8          = NULL;
    ls.vt.factor_chol_csc_d_i8_destroy = NULL;
    ls.vt.factor_chol_csc_z_i8        = NULL;
    ls.vt.solve_chol_csc_z_i8          = NULL;
    ls.vt.factor_chol_csc_z_i8_destroy = NULL;
    ls.vt.refactor_csc_d_i8           = NULL;

    return ls;
}

#else  /* !MATX_HAVE_AGMG */

/* Stub: AGMG not available at build time */
matx_sparse_linsolve_t matx_linsolve_make_agmg(matx_alloc_t alloc)
{
    matx_sparse_linsolve_t ls;
    memset(&ls, 0, sizeof(ls));
    ls.kind  = MATX_LINSOLVE_BACKEND_AGMG;
    ls.alloc = alloc;
    /* all vtable entries NULL → operations will return NOT_SUPPORTED */
    return ls;
}

#endif /* MATX_HAVE_AGMG */