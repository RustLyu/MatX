#include "matx/matx_sparse_solve.h"

// Forward decls
matx_sparse_linsolve_t matx_linsolve_make_suitesparse_klu(void);
matx_sparse_linsolve_t matx_linsolve_make_umfpack(void);
matx_sparse_linsolve_t matx_linsolve_make_cxsparse(void);
matx_sparse_linsolve_t matx_linsolve_make_superlu(void);
matx_sparse_linsolve_t matx_linsolve_make_mumps(void);

const char* matx_sparse_linsolve_backend_name(matx_sparse_linsolve_backend_kind_t k) {
  switch (k) {
    case MATX_LINSOLVE_BACKEND_SUITESPARSE_KLU: return "SUITESPARSE";
    case MATX_LINSOLVE_BACKEND_UMFPACK: return "UMFPACK";
    case MATX_LINSOLVE_BACKEND_CXSPARSE: return "CXSPARSE";
    case MATX_LINSOLVE_BACKEND_SUPERLU: return "SUPERLU";
    case MATX_LINSOLVE_BACKEND_MUMPS: return "MUMPS";
    default: return "UNKNOWN";
  }
}

matx_sparse_linsolve_t matx_sparse_linsolve_default(void) {
#if MATX_HAVE_UMFPACK
    return matx_linsolve_make_mumps();
    //return matx_linsolve_make_umfpack();
#elif MATX_HAVE_CXSPARSE
  return matx_linsolve_make_cxsparse();
#elif MATX_HAVE_SUPERLU
  return matx_linsolve_make_superlu();
#elif MATX_HAVE_MUMPS
  return matx_linsolve_make_mumps();
#else
  return matx_linsolve_make_suitesparse();
#endif
}

matx_sparse_linsolve_t matx_sparse_linsolve_by_type(matx_sparse_linsolve_backend_kind_t k) {
  switch (k) {
    case MATX_LINSOLVE_BACKEND_UMFPACK: return matx_linsolve_make_umfpack();
    case MATX_LINSOLVE_BACKEND_CXSPARSE: return matx_linsolve_make_cxsparse();
    case MATX_LINSOLVE_BACKEND_SUPERLU: return matx_linsolve_make_superlu();
    case MATX_LINSOLVE_BACKEND_MUMPS: return matx_linsolve_make_mumps();
    case MATX_LINSOLVE_BACKEND_SUITESPARSE_KLU:
    default: return matx_linsolve_make_suitesparse_klu();
  }
}

// High-level wrappers ------------------------------------------------------

// Sparse real
matx_status_t matx_factor_csc_f64(const matx_sparse_linsolve_t* ls,
                                  matx_coo_f64_t A,
                                  matx_factor_sparse_f64_t* out_F) {
  if (!ls || !A || !out_F) return MATX_ERR_INVALID_ARG;
  if (!ls->vt.factor_csc_f64) return MATX_ERR_NOT_SUPPORTED;
  return ls->vt.factor_csc_f64(A, out_F);
}

matx_status_t matx_solve_csc_f64_factor(const matx_sparse_linsolve_t* ls,
                                        matx_factor_sparse_f64_t* F,
                                        const matx_double* b,
    matx_double* x) {
  if (!ls || !F || !b || !x) return MATX_ERR_INVALID_ARG;
  if (!ls->vt.solve_csc_f64) return MATX_ERR_NOT_SUPPORTED;
  return ls->vt.solve_csc_f64(F, b, x);
}

void matx_factor_csc_f64_destroy(const matx_sparse_linsolve_t* ls,
                                 matx_factor_sparse_f64_t* F) {
  if (!ls || !F) return;
  if (ls->vt.factor_csc_f64_destroy) {
    ls->vt.factor_csc_f64_destroy(F);
  }
}

matx_status_t matx_solve_csc_f64(const matx_sparse_linsolve_t* ls,
                                 matx_coo_f64_t A,
                                 const matx_double* b,
    matx_double* x) {
  if (!ls || !A || !b || !x) return MATX_ERR_INVALID_ARG;
  matx_factor_sparse_f64_t F;
  F.reserved = NULL;
  matx_status_t st = matx_factor_csc_f64(ls, A, &F);
  if (st != MATX_OK) 
      return st;
  st = matx_solve_csc_f64_factor(ls, &F, b, x);
  matx_factor_csc_f64_destroy(ls, &F);
  return st;
}


// Complex variants (default to NOT_SUPPORTED until backend provides them)
matx_status_t matx_factor_csc_c64(const matx_sparse_linsolve_t* ls,
                                  matx_coo_c64_t A,
                                  matx_factor_sparse_c64_t* out_F) {
  if (!ls || !A || !out_F) return MATX_ERR_INVALID_ARG;
  if (!ls->vt.factor_csc_c64) return MATX_ERR_NOT_SUPPORTED;
  return ls->vt.factor_csc_c64(A, out_F);
}

matx_status_t matx_solve_csc_c64_factor(const matx_sparse_linsolve_t* ls,
                                        matx_factor_sparse_c64_t* F,
                                        const matx_vec_c64_t b,
                                        matx_vec_c64_t x) {
  if (!ls || !F || !b || !x) return MATX_ERR_INVALID_ARG;
  if (!ls->vt.solve_csc_c64) return MATX_ERR_NOT_SUPPORTED;
  return ls->vt.solve_csc_c64(F, b, x);
}

void matx_factor_csc_c64_destroy(const matx_sparse_linsolve_t* ls,
                                 matx_factor_sparse_c64_t* F) {
  if (!ls || !F) 
      return;
  if (ls->vt.factor_csc_c64_destroy) {
    ls->vt.factor_csc_c64_destroy(F);
  }
}

matx_status_t matx_solve_csc_c64(const matx_sparse_linsolve_t* ls,
                                 matx_coo_c64_t A,
                                 const matx_vec_c64_t b,
                                 matx_vec_c64_t x) {
  if (!ls || !A || !b || !x) 
      return MATX_ERR_INVALID_ARG;
  matx_factor_sparse_c64_t F;
  matx_status_t st = matx_factor_csc_c64(ls, A, &F);
  if (st != MATX_OK) 
      return st;
  st = matx_solve_csc_c64_factor(ls, &F, b, x);
  matx_factor_csc_c64_destroy(ls, &F);
  return st;
}

