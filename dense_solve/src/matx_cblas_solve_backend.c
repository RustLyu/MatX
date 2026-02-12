#include "matx/matx_dense_solve.h"

// Forward decls
matx_dense_linsolve_t matx_dense_linsolve_make_cblas(void);

const char* matx_dense_linsolve_backend_name(matx_dense_linsolve_backend_kind_t k) {
  switch (k) {
    case MATX_LINSOLVE_BACKEND_CBLAS: return "CBLAS||BLIS";
    default: return "UNKNOWN";
  }
}

matx_dense_linsolve_t matx_dense_linsolve_default(void) {
  return matx_dense_linsolve_make_cblas();
}

// High-level wrappers ------------------------------------------------------

// Dense real
matx_status_t matx_factor_dense_f64(const matx_dense_linsolve_t* ls,
                                    const matx_dense_f64_t* A,
                                    matx_factor_dense_f64_t** out_F) {
  if (!ls || !A || !out_F) return MATX_ERR_INVALID_ARG;
  if (!ls->vt.factor_dense_f64) return MATX_ERR_NOT_SUPPORTED;
  return ls->vt.factor_dense_f64(A, out_F);
}

matx_status_t matx_solve_dense_f64_factor(const matx_dense_linsolve_t* ls,
                                          const matx_factor_dense_f64_t* F,
                                          const matx_double* b,
    matx_double* x) {
  if (!ls || !F || !b || !x) return MATX_ERR_INVALID_ARG;
  if (!ls->vt.solve_dense_f64) return MATX_ERR_NOT_SUPPORTED;
  return ls->vt.solve_dense_f64(F, b, x);
}

void matx_factor_dense_f64_destroy(const matx_dense_linsolve_t* ls,
                                   matx_factor_dense_f64_t* F) {
  if (!ls || !F) return;
  if (ls->vt.factor_dense_f64_destroy) {
    ls->vt.factor_dense_f64_destroy(F);
  }
}

matx_status_t matx_solve_dense_f64(const matx_dense_linsolve_t* ls,
                                   const matx_dense_f64_t* A,
                                   const matx_double* b,
    matx_double* x) {
  if (!ls || !A || !b || !x) return MATX_ERR_INVALID_ARG;
  matx_factor_dense_f64_t* F = NULL;
  matx_status_t st = matx_factor_dense_f64(ls, A, &F);
  if (st != MATX_OK) return st;
  st = matx_solve_dense_f64_factor(ls, F, b, x);
  matx_factor_dense_f64_destroy(ls, F);
  return st;
}

matx_status_t matx_factor_dense_c64(const matx_dense_linsolve_t* ls,
                                    const matx_dense_c64_t* A,
                                    matx_factor_dense_c64_t** out_F) {
  if (!ls || !A || !out_F) return MATX_ERR_INVALID_ARG;
  if (!ls->vt.factor_dense_c64) return MATX_ERR_NOT_SUPPORTED;
  return ls->vt.factor_dense_c64(A, out_F);
}

matx_status_t matx_solve_dense_c64_factor(const matx_dense_linsolve_t* ls,
                                          const matx_factor_dense_c64_t* F,
                                          const matx_vec_c64_t* b,
                                          matx_vec_c64_t* x) {
  if (!ls || !F || !b || !x) return MATX_ERR_INVALID_ARG;
  if (!ls->vt.solve_dense_c64) return MATX_ERR_NOT_SUPPORTED;
  return ls->vt.solve_dense_c64(F, b, x);
}

void matx_factor_dense_c64_destroy(const matx_dense_linsolve_t* ls,
                                   matx_factor_dense_c64_t* F) {
  if (!ls || !F) return;
  if (ls->vt.factor_dense_c64_destroy) {
    ls->vt.factor_dense_c64_destroy(F);
  }
}

matx_status_t matx_solve_dense_c64(const matx_dense_linsolve_t* ls,
                                   const matx_dense_c64_t* A,
                                   const matx_vec_c64_t* b,
                                   matx_vec_c64_t* x) {
  if (!ls || !A || !b || !x) return MATX_ERR_INVALID_ARG;
  matx_factor_dense_c64_t* F = NULL;
  matx_status_t st = matx_factor_dense_c64(ls, A, &F);
  if (st != MATX_OK) return st;
  st = matx_solve_dense_c64_factor(ls, F, b, x);
  matx_factor_dense_c64_destroy(ls, F);
  return st;
}

