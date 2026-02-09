#include "matx/matx_solve.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "suitesparse/klu.h"

struct matx_factor_sparse_f64_t {
  klu_l_symbolic* S;
  klu_l_numeric* N;
  klu_l_common common;
  matx_uint64_t n;
};

// Dense factorization (simple LU in C for now)
struct matx_factor_dense_f64_t {
  matx_uint64_t n;
  matx_double* lu; // column-major, combined L+U
  matx_uint64_t* piv;   // pivot indices, size n
};

// Sparse real: KLU-based ---------------------------------------------------
static matx_status_t ss_factor_csc_f64(const matx_csc_f64_t* A,
                                       matx_factor_sparse_f64_t** out_F) {
  if (!A || !out_F) return MATX_ERR_INVALID_ARG;
  if (!A->col_ptr || !A->row_ind || !A->values) return MATX_ERR_INVALID_ARG;
  if (A->nrows != A->ncols) return MATX_ERR_INVALID_ARG;

  const matx_uint64_t n = A->nrows;
  if (n == 0) return MATX_ERR_INVALID_ARG;
  if (n > (matx_uint64_t)INT_MAX) return MATX_ERR_NOT_SUPPORTED;

  matx_factor_sparse_f64_t* F =
      (matx_factor_sparse_f64_t*)malloc(sizeof(*F));
  if (!F) return MATX_ERR_OUT_OF_MEMORY;

  memset(F, 0, sizeof(*F));
  F->n = (matx_uint64_t)n;
  klu_l_defaults(&F->common);

  F->S = klu_l_analyze(F->n,
                     A->col_ptr,
                     A->row_ind,
                     &F->common);
  if (!F->S) {
    free(F);
    return MATX_ERR_INTERNAL;
  }

  matx_double* Ax = (matx_double*)malloc(A->nnz * sizeof(matx_double));
  if (!Ax) {
    klu_free_symbolic(&F->S, &F->common);
    free(F);
    return MATX_ERR_OUT_OF_MEMORY;
  }
  memcpy(Ax, A->values, A->nnz * sizeof(matx_double));

  F->N = klu_l_factor(A->col_ptr,
                    A->row_ind,
                    Ax,
                    F->S,
                    &F->common);
  free(Ax);

  if (!F->N) {
    klu_free_symbolic(&F->S, &F->common);
    free(F);
    return MATX_ERR_INTERNAL;
  }

  *out_F = F;
  return MATX_OK;
}

static matx_status_t ss_solve_csc_f64(const matx_factor_sparse_f64_t* F,
                                      const matx_double* b,
                                        matx_double* x) {
  if (!F || !b || !x) return MATX_ERR_INVALID_ARG;

  const matx_uint64_t n = F->n;
  for (matx_uint64_t i = 0; i < n; ++i) {
    x[i] = b[i];
  }

  const int status = klu_l_solve(F->S, F->N, n, 1, x, &F->common);
  if (!status) return MATX_ERR_INTERNAL;
  return MATX_OK;
}

static void ss_factor_csc_f64_destroy(matx_factor_sparse_f64_t* F) {
  if (!F) return;
  klu_l_free_numeric(&F->N, &F->common);
  klu_l_free_symbolic(&F->S, &F->common);
  free(F);
}


static matx_status_t ss_factor_csc_c64(
    const matx_csc_c64_t* A,
    matx_factor_sparse_c64_t** out_F)
{
    if (!A || !out_F) return MATX_ERR_INVALID_ARG;
    if (!A->col_ptr || !A->row_ind || !A->values)
        return MATX_ERR_INVALID_ARG;
    if (A->nrows != A->ncols) return MATX_ERR_INVALID_ARG;

    const matx_uint64_t n = A->nrows;
    if (n > INT_MAX) return MATX_ERR_NOT_SUPPORTED;

    matx_factor_sparse_c64_t* F =
        (matx_factor_sparse_c64_t*)malloc(sizeof(*F));
    if (!F) return MATX_ERR_OUT_OF_MEMORY;

    memset(F, 0, sizeof(*F));
    F->n = (matx_uint64_t)n;

    klu_l_defaults(&F->common);

    F->S = klu_l_analyze(
        F->n,
        A->col_ptr,
        A->row_ind,
        &F->common);

    if (!F->S) goto fail;

    /* KLU 会修改 Ax，需要复制 */
    void* Ax = malloc(A->nnz * sizeof(matx_double) * 2);
    if (!Ax) goto fail;

    memcpy(Ax, A->values, A->nnz * sizeof(matx_double) * 2);

    F->N = klu_zl_factor(
        A->col_ptr,
        A->row_ind,
        Ax,
        F->S,
        &F->common);

    free(Ax);

    if (!F->N) goto fail;

    *out_F = F;
    return MATX_OK;

fail:
    klu_free_symbolic(&F->S, &F->common);
    free(F);
    return MATX_ERR_INTERNAL;
}

static matx_status_t ss_solve_csc_c64(
    const matx_factor_sparse_c64_t* F,
    const matx_vec_c64_t* b,
    matx_vec_c64_t* x)
{
    if (!F || !b || !x) return MATX_ERR_INVALID_ARG;

    matx_uint64_t n = F->n;

    memcpy(x->data, b->data, sizeof(matx_double) * 2 * n);

    matx_uint64_t status = klu_zl_solve(
        F->S, F->N, n, 1, x->data, &F->common);

    if (!status) return MATX_ERR_INTERNAL;

    return MATX_OK;
}

static void ss_factor_csc_c64_destroy(
    matx_factor_sparse_c64_t* F)
{
    if (!F) return;

    klu_free_numeric(&F->N, &F->common);
    klu_free_symbolic(&F->S, &F->common);
    free(F);
}

// Dense real: LU + solve using LAPACK when available -----------------------
static matx_status_t ss_factor_dense_f64(const matx_dense_f64_t* A,
                                         matx_factor_dense_f64_t** out_F) {
  if (!A || !out_F) return MATX_ERR_INVALID_ARG;
  if (A->rows != A->cols) return MATX_ERR_INVALID_ARG;
  if (A->layout != MATX_COL_MAJOR) return MATX_ERR_NOT_SUPPORTED; // simplify: col-major only

  const size_t n = A->rows;
  matx_factor_dense_f64_t* F =
      (matx_factor_dense_f64_t*)malloc(sizeof(*F));
  if (!F) return MATX_ERR_OUT_OF_MEMORY;
  memset(F, 0, sizeof(*F));
  F->n = n;

  F->lu = (matx_double*)malloc(n * n * sizeof(matx_double));
  F->piv = (matx_uint64_t*)malloc(n * sizeof(matx_uint64_t));
  if (!F->lu || !F->piv) {
    free(F->lu);
    free(F->piv);
    free(F);
    return MATX_ERR_OUT_OF_MEMORY;
  }

  // Copy A into lu (column-major)
  for (size_t j = 0; j < n; ++j) {
    for (size_t i = 0; i < n; ++i) {
      F->lu[i + j * n] = A->data[i + j * A->stride];
    }
  }
  for (size_t i = 0; i < n; ++i) F->piv[i] = 0;

  matx_uint64_t N = (matx_uint64_t)n;
  matx_uint64_t lda = (matx_uint64_t)n;
  matx_uint64_t info = 0;

  dgetrf_(&N, &N, F->lu, &lda, F->piv, &info);
  if (info != 0) {
    free(F->lu);
    free(F->piv);
    free(F);
    return MATX_ERR_INTERNAL;
  }

  *out_F = F;
  return MATX_OK;
}

static matx_status_t ss_solve_dense_f64(const matx_factor_dense_f64_t* F,
                                        const matx_double* b,
    matx_double* x) {
  if (!F || !b || !x) return MATX_ERR_INVALID_ARG;
  const matx_uint64_t n = F->n;
  // Copy b into x
  for (matx_uint64_t i = 0; i < n; ++i) {
    x[i] = b[i];
  }

  matx_uint64_t N = (int)n;
  matx_uint64_t nrhs = 1;
  matx_uint64_t lda = (int)n;
  matx_uint64_t ldb = (int)n;
  matx_uint64_t info = 0;
  char trans = 'N';

  dgetrs_(&trans, &N, &nrhs, F->lu, &lda, F->piv, x, &ldb, &info);
  if (info != 0) {
    return MATX_ERR_INTERNAL;
  }

  return MATX_OK;
}

static void ss_factor_dense_f64_destroy(matx_factor_dense_f64_t* F) {
  if (!F) return;
  free(F->lu);
  free(F->piv);
  free(F);
}



static matx_status_t ss_factor_dense_c64(
    const matx_dense_c64_t* A,
    matx_factor_dense_c64_t** out_F)
{
#if !(defined(MATX_HAVE_OPENBLAS) || defined(MATX_HAVE_BLIS))
    return MATX_ERR_NOT_SUPPORTED;
#else
    if (!A || !out_F) return MATX_ERR_INVALID_ARG;
    if (A->rows != A->cols) return MATX_ERR_INVALID_ARG;
    if (A->layout != MATX_COL_MAJOR)
        return MATX_ERR_NOT_SUPPORTED;

    matx_uint64_t n = A->rows;

    matx_factor_dense_c64_t* F =
        malloc(sizeof(*F));
    if (!F) return MATX_ERR_OUT_OF_MEMORY;

    F->n = n;

    F->lu = malloc(sizeof(matx_double) * 2 * n * n);
    F->piv = malloc(sizeof(matx_uint64_t) * n);

    if (!F->lu || !F->piv) goto fail;

    /* copy matrix */
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i)
            memcpy(&F->lu[2 * (i + j * n)],
                &A->data[2 * (i + j * A->stride)],
                sizeof(matx_double) * 2);

    matx_uint64_t N = (matx_uint64_t)n;
    matx_uint64_t lda = (matx_uint64_t)n;
    matx_uint64_t info = 0;

    zgetrf_(&N, &N, F->lu, &lda, F->piv, &info);

    if (info != 0) goto fail;

    *out_F = F;
    return MATX_OK;

fail:
    free(F->lu);
    free(F->piv);
    free(F);
    return MATX_ERR_INTERNAL;
#endif
}

static matx_status_t ss_solve_dense_c64(
    const matx_factor_dense_c64_t* F,
    const matx_vec_c64_t* b,
    matx_vec_c64_t* x)
{
    if (!F || !b || !x) return MATX_ERR_INVALID_ARG;

    size_t n = F->n;

    memcpy(x->data, b->data, sizeof(matx_double) * 2 * n);

    matx_uint64_t N = (matx_uint64_t)n;
    matx_uint64_t nrhs = 1;
    matx_uint64_t lda = (matx_uint64_t)n;
    matx_uint64_t ldb = (matx_uint64_t)n;
    matx_uint64_t info = 0;
    char trans = 'N';

    zgetrs_(&trans, &N, &nrhs,
        F->lu, &lda, F->piv,
        x->data, &ldb,
        &info);

    if (info != 0) return MATX_ERR_INTERNAL;

    return MATX_OK;
}

static void ss_factor_dense_c64_destroy(
    matx_factor_dense_c64_t* F)
{
    if (!F) return;
    free(F->lu);
    free(F->piv);
    free(F);
}


matx_linsolve_t matx_linsolve_make_suitesparse(void) {
  matx_linsolve_t ls;
  ls.kind = MATX_LINSOLVE_BACKEND_SUITESPARSE;

  ls.vt.factor_csc_f64 = &ss_factor_csc_f64;
  ls.vt.solve_csc_f64 = &ss_solve_csc_f64;
  ls.vt.factor_csc_f64_destroy = &ss_factor_csc_f64_destroy;

  ls.vt.factor_dense_f64 = &ss_factor_dense_f64;
  ls.vt.solve_dense_f64 = &ss_solve_dense_f64;
  ls.vt.factor_dense_f64_destroy = &ss_factor_dense_f64_destroy;

  ls.vt.factor_csc_c64 = &ss_factor_csc_c64;
  ls.vt.solve_csc_c64 = &ss_solve_csc_c64;
  ls.vt.factor_csc_c64_destroy = &ss_factor_csc_c64_destroy;

  ls.vt.factor_dense_c64 = &ss_factor_dense_c64;
  ls.vt.solve_dense_c64 = &ss_solve_dense_c64;
  ls.vt.factor_dense_c64_destroy = &ss_factor_dense_c64_destroy;

  return ls;
}