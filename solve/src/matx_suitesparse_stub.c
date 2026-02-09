#include "matx/matx_solve.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "suitesparse/klu.h"

struct matx_factor_sparse_f64_t {
  klu_symbolic* S;
  klu_numeric* N;
  klu_common common;
  int n;
};

// Dense factorization (simple LU in C for now)
struct matx_factor_dense_f64_t {
  size_t n;
  double* lu; // column-major, combined L+U
  int* piv;   // pivot indices, size n
};

// Sparse real: KLU-based ---------------------------------------------------
static matx_status_t ss_factor_csc_f64(const matx_csc_f64_t* A,
                                       matx_factor_sparse_f64_t** out_F) {
  if (!A || !out_F) return MATX_ERR_INVALID_ARG;
  if (!A->col_ptr || !A->row_ind || !A->values) return MATX_ERR_INVALID_ARG;
  if (A->nrows != A->ncols) return MATX_ERR_INVALID_ARG;

  const size_t n = A->nrows;
  if (n == 0) return MATX_ERR_INVALID_ARG;
  if (n > (size_t)INT_MAX) return MATX_ERR_NOT_SUPPORTED;

  matx_factor_sparse_f64_t* F =
      (matx_factor_sparse_f64_t*)malloc(sizeof(*F));
  if (!F) return MATX_ERR_OUT_OF_MEMORY;

  memset(F, 0, sizeof(*F));
  F->n = (int)n;
  klu_defaults(&F->common);

  F->S = klu_analyze(F->n,
                     (int*)A->col_ptr,
                     (int*)A->row_ind,
                     &F->common);
  if (!F->S) {
    free(F);
    return MATX_ERR_INTERNAL;
  }

  double* Ax = (double*)malloc(A->nnz * sizeof(double));
  if (!Ax) {
    klu_free_symbolic(&F->S, &F->common);
    free(F);
    return MATX_ERR_OUT_OF_MEMORY;
  }
  memcpy(Ax, A->values, A->nnz * sizeof(double));

  F->N = klu_factor((int*)A->col_ptr,
                    (int*)A->row_ind,
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
                                      const double* b,
                                      double* x) {
  if (!F || !b || !x) return MATX_ERR_INVALID_ARG;

  const int n = F->n;
  for (int i = 0; i < n; ++i) {
    x[i] = b[i];
  }

  const int status = klu_solve(F->S, F->N, n, 1, x, &F->common);
  if (!status) return MATX_ERR_INTERNAL;
  return MATX_OK;
}

static void ss_factor_csc_f64_destroy(matx_factor_sparse_f64_t* F) {
  if (!F) return;
  klu_free_numeric(&F->N, &F->common);
  klu_free_symbolic(&F->S, &F->common);
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

    const size_t n = A->nrows;
    if (n > INT_MAX) return MATX_ERR_NOT_SUPPORTED;

    matx_factor_sparse_c64_t* F =
        (matx_factor_sparse_c64_t*)malloc(sizeof(*F));
    if (!F) return MATX_ERR_OUT_OF_MEMORY;

    memset(F, 0, sizeof(*F));
    F->n = (int)n;

    klu_defaults(&F->common);

    F->S = klu_analyze(
        F->n,
        (int*)A->col_ptr,
        (int*)A->row_ind,
        &F->common);

    if (!F->S) goto fail;

    /* KLU 会修改 Ax，需要复制 */
    void* Ax = malloc(A->nnz * sizeof(double) * 2);
    if (!Ax) goto fail;

    memcpy(Ax, A->values, A->nnz * sizeof(double) * 2);

    F->N = klu_z_factor(
        (int*)A->col_ptr,
        (int*)A->row_ind,
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

    int n = F->n;

    memcpy(x->data, b->data, sizeof(double) * 2 * n);

    int status = klu_z_solve(
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

#if defined(MATX_HAVE_OPENBLAS) || defined(MATX_HAVE_BLIS)
// Use LAPACK dgetrf/dgetrs from BLAS/LAPACK (OpenBLAS or vendor BLAS)
extern void dgetrf_(const int* m, const int* n,
                    double* a, const int* lda,
                    int* ipiv, int* info);

extern void dgetrs_(const char* trans, const int* n, const int* nrhs,
                    const double* a, const int* lda,
                    const int* ipiv,
                    double* b, const int* ldb,
                    int* info);
extern void zgetrf_(
    const int* m,
    const int* n,
    double* a,
    const int* lda,
    int* ipiv,
    int* info);

extern void zgetrs_(
    const char* trans,
    const int* n,
    const int* nrhs,
    const double* a,
    const int* lda,
    const int* ipiv,
    double* b,
    const int* ldb,
    int* info);
#endif

// Dense real: LU + solve using LAPACK when available -----------------------
static matx_status_t ss_factor_dense_f64(const matx_dense_f64_t* A,
                                         matx_factor_dense_f64_t** out_F) {
  if (!A || !out_F) return MATX_ERR_INVALID_ARG;
  if (A->rows != A->cols) return MATX_ERR_INVALID_ARG;
#if !(defined(MATX_HAVE_OPENBLAS) || defined(MATX_HAVE_BLIS))
  (void)A;
  (void)out_F;
  return MATX_ERR_NOT_SUPPORTED;
#else
  if (A->layout != MATX_COL_MAJOR) return MATX_ERR_NOT_SUPPORTED; // simplify: col-major only

  const size_t n = A->rows;
  matx_factor_dense_f64_t* F =
      (matx_factor_dense_f64_t*)malloc(sizeof(*F));
  if (!F) return MATX_ERR_OUT_OF_MEMORY;
  memset(F, 0, sizeof(*F));
  F->n = n;

  F->lu = (double*)malloc(n * n * sizeof(double));
  F->piv = (int*)malloc(n * sizeof(int));
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

  int N = (int)n;
  int lda = (int)n;
  int info = 0;

  dgetrf_(&N, &N, F->lu, &lda, F->piv, &info);
  if (info != 0) {
    free(F->lu);
    free(F->piv);
    free(F);
    return MATX_ERR_INTERNAL;
  }

  *out_F = F;
  return MATX_OK;
#endif
}

static matx_status_t ss_solve_dense_f64(const matx_factor_dense_f64_t* F,
                                        const double* b,
                                        double* x) {
  if (!F || !b || !x) return MATX_ERR_INVALID_ARG;
  const size_t n = F->n;
#if !(defined(MATX_HAVE_OPENBLAS) || defined(MATX_HAVE_BLIS))
  (void)n;
  (void)F;
  (void)b;
  (void)x;
  return MATX_ERR_NOT_SUPPORTED;
#else
  // Copy b into x
  for (size_t i = 0; i < n; ++i) {
    x[i] = b[i];
  }

  int N = (int)n;
  int nrhs = 1;
  int lda = (int)n;
  int ldb = (int)n;
  int info = 0;
  char trans = 'N';

  dgetrs_(&trans, &N, &nrhs, F->lu, &lda, F->piv, x, &ldb, &info);
  if (info != 0) {
    return MATX_ERR_INTERNAL;
  }

  return MATX_OK;
#endif
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

    size_t n = A->rows;

    matx_factor_dense_c64_t* F =
        malloc(sizeof(*F));
    if (!F) return MATX_ERR_OUT_OF_MEMORY;

    F->n = n;

    F->lu = malloc(sizeof(double) * 2 * n * n);
    F->piv = malloc(sizeof(int) * n);

    if (!F->lu || !F->piv) goto fail;

    /* copy matrix */
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i)
            memcpy(&F->lu[2 * (i + j * n)],
                &A->data[2 * (i + j * A->stride)],
                sizeof(double) * 2);

    int N = (int)n;
    int lda = (int)n;
    int info = 0;

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

    memcpy(x->data, b->data, sizeof(double) * 2 * n);

    int N = (int)n;
    int nrhs = 1;
    int lda = (int)n;
    int ldb = (int)n;
    int info = 0;
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