#include "matx/matx_dense_solve.h"
#include "matx/matx_log.h"

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
matx_status_t matx_factor_dense_d_i8(const matx_dense_linsolve_t* ls,
                                    const matx_dense_d_i8_t A,
                                    matx_factor_dense_d_i8_t** out_F) {
  if (!ls || !A || !out_F) {
  	MATX_ERROR("%s: invalid argument", __func__);
  	return MATX_ERR_INVALID_ARG;
  }
  if (!ls->vt.factor_dense_d_i8) {
  	MATX_ERROR("%s: operation not supported", __func__);
  	return MATX_ERR_NOT_SUPPORTED;
  }
  return ls->vt.factor_dense_d_i8(A, out_F);
}

matx_status_t matx_solve_dense_d_i8_factor(const matx_dense_linsolve_t* ls,
                                          const matx_factor_dense_d_i8_t* F,
                                          const matx_double* b,
    matx_double* x) {
  if (!ls || !F || !b || !x) {
  	MATX_ERROR("%s: invalid argument", __func__);
  	return MATX_ERR_INVALID_ARG;
  }
  if (!ls->vt.solve_dense_d_i8) {
  	MATX_ERROR("%s: operation not supported", __func__);
  	return MATX_ERR_NOT_SUPPORTED;
  }
  return ls->vt.solve_dense_d_i8(F, b, x);
}

void matx_factor_dense_d_i8_destroy(const matx_dense_linsolve_t* ls,
                                   matx_factor_dense_d_i8_t* F) {
  if (!ls || !F) return;
  if (ls->vt.factor_dense_d_i8_destroy) {
    ls->vt.factor_dense_d_i8_destroy(F);
  }
}

matx_status_t matx_solve_dense_d_i8(const matx_dense_linsolve_t* ls,
                                   const matx_dense_d_i8_t A,
                                   const matx_double* b,
    matx_double* x) {
  if (!ls || !A || !b || !x) {
  	MATX_ERROR("%s: invalid argument", __func__);
  	return MATX_ERR_INVALID_ARG;
  }
  matx_factor_dense_d_i8_t* F = NULL;
  matx_status_t st = matx_factor_dense_d_i8(ls, A, &F);
  if (st != MATX_OK) return st;
  st = matx_solve_dense_d_i8_factor(ls, F, b, x);
  matx_factor_dense_d_i8_destroy(ls, F);
  return st;
}

matx_status_t matx_factor_dense_z_i8(const matx_dense_linsolve_t* ls,
                                    const matx_dense_z_i8_t A,
                                    matx_factor_dense_z_i8_t** out_F) {
  if (!ls || !A || !out_F) {
  	MATX_ERROR("%s: invalid argument", __func__);
  	return MATX_ERR_INVALID_ARG;
  }
  if (!ls->vt.factor_dense_z_i8) {
  	MATX_ERROR("%s: operation not supported", __func__);
  	return MATX_ERR_NOT_SUPPORTED;
  }
  return ls->vt.factor_dense_z_i8(A, out_F);
}

matx_status_t matx_solve_dense_z_i8_factor(const matx_dense_linsolve_t* ls,
                                          const matx_factor_dense_z_i8_t* F,
                                          const matx_vec_z_i8_t b,
                                          matx_vec_z_i8_t x) {
  if (!ls || !F || !b || !x) {
  	MATX_ERROR("%s: invalid argument", __func__);
  	return MATX_ERR_INVALID_ARG;
  }
  if (!ls->vt.solve_dense_z_i8) {
  	MATX_ERROR("%s: operation not supported", __func__);
  	return MATX_ERR_NOT_SUPPORTED;
  }
  return ls->vt.solve_dense_z_i8(F, b, x);
}

void matx_factor_dense_z_i8_destroy(const matx_dense_linsolve_t* ls,
                                   matx_factor_dense_z_i8_t* F) {
  if (!ls || !F) return;
  if (ls->vt.factor_dense_z_i8_destroy) {
    ls->vt.factor_dense_z_i8_destroy(F);
  }
}

matx_status_t matx_solve_dense_z_i8(const matx_dense_linsolve_t* ls,
                                   const matx_dense_z_i8_t A,
                                   const matx_vec_z_i8_t b,
                                   matx_vec_z_i8_t x) {
  if (!ls || !A || !b || !x) {
  	MATX_ERROR("%s: invalid argument", __func__);
  	return MATX_ERR_INVALID_ARG;
  }
  matx_factor_dense_z_i8_t* F = NULL;
  matx_status_t st = matx_factor_dense_z_i8(ls, A, &F);
  if (st != MATX_OK) return st;
  st = matx_solve_dense_z_i8_factor(ls, F, b, x);
  matx_factor_dense_z_i8_destroy(ls, F);
  return st;
}

// ---- Cholesky wrappers ----

matx_status_t matx_factor_chol_d_i8(const matx_dense_linsolve_t* ls,
	const matx_dense_d_i8_t A, matx_uplo_t uplo, matx_factor_dense_d_i8_t** out_F)
{
	if (!ls || !A || !out_F) {
		MATX_ERROR("%s: invalid argument", __func__);
		return MATX_ERR_INVALID_ARG;
	}
	if (!ls->vt.potrf_d_i8) {
		MATX_ERROR("%s: operation not supported", __func__);
		return MATX_ERR_NOT_SUPPORTED;
	}
	return ls->vt.potrf_d_i8(A, uplo, out_F);
}

matx_status_t matx_solve_chol_d_i8(const matx_dense_linsolve_t* ls,
	const matx_factor_dense_d_i8_t* F, const matx_double* b, matx_double* x)
{
	if (!ls || !F || !b || !x) {
		MATX_ERROR("%s: invalid argument", __func__);
		return MATX_ERR_INVALID_ARG;
	}
	if (!ls->vt.potrs_d_i8) {
		MATX_ERROR("%s: operation not supported", __func__);
		return MATX_ERR_NOT_SUPPORTED;
	}
	return ls->vt.potrs_d_i8(F, b, x);
}

matx_status_t matx_solve_chol_d_i8_oneshot(const matx_dense_linsolve_t* ls,
	const matx_dense_d_i8_t A, matx_uplo_t uplo, const matx_double* b, matx_double* x) 
{
	if (!ls || !A || !b || !x) {
		MATX_ERROR("%s: invalid argument", __func__);
		return MATX_ERR_INVALID_ARG;
	}
	matx_factor_dense_d_i8_t* F = NULL;
	matx_status_t st = matx_factor_chol_d_i8(ls, A, uplo, &F);
	if (st != MATX_OK) 
		return st;
	st = matx_solve_chol_d_i8(ls, F, b, x);
	matx_factor_dense_d_i8_destroy(ls, F);
	return st;
}

matx_status_t matx_factor_chol_z_i8(const matx_dense_linsolve_t* ls,
	const matx_dense_z_i8_t A, matx_uplo_t uplo, matx_factor_dense_z_i8_t** out_F) 
{
	if (!ls || !A || !out_F) {
		MATX_ERROR("%s: invalid argument", __func__);
		return MATX_ERR_INVALID_ARG;
	}
	if (!ls->vt.potrf_z_i8) {
		MATX_ERROR("%s: operation not supported", __func__);
		return MATX_ERR_NOT_SUPPORTED;
	}
	return ls->vt.potrf_z_i8(A, uplo, out_F);
}

matx_status_t matx_solve_chol_z_i8(const matx_dense_linsolve_t* ls,
	const matx_factor_dense_z_i8_t* F, const matx_vec_z_i8_t b, matx_vec_z_i8_t x) 
{
	if (!ls || !F || !b || !x) {
		MATX_ERROR("%s: invalid argument", __func__);
		return MATX_ERR_INVALID_ARG;
	}
	if (!ls->vt.potrs_z_i8) {
		MATX_ERROR("%s: operation not supported", __func__);
		return MATX_ERR_NOT_SUPPORTED;
	}
	return ls->vt.potrs_z_i8(F, b, x);
}

// ---- GELS wrappers ----

matx_status_t matx_gels_d_i8(const matx_dense_linsolve_t* ls,
	const matx_dense_d_i8_t A, const matx_double* b, matx_double* x) 
{
	if (!ls || !A || !b || !x) {
		MATX_ERROR("%s: invalid argument", __func__);
		return MATX_ERR_INVALID_ARG;
	}
	if (!ls->vt.gels_d_i8) {
		MATX_ERROR("%s: operation not supported", __func__);
		return MATX_ERR_NOT_SUPPORTED;
	}
	return ls->vt.gels_d_i8(A, b, x);
}

matx_status_t matx_gels_z_i8(const matx_dense_linsolve_t* ls,
	const matx_dense_z_i8_t A, const matx_vec_z_i8_t b, matx_vec_z_i8_t x) 
{
	if (!ls || !A || !b || !x) {
		MATX_ERROR("%s: invalid argument", __func__);
		return MATX_ERR_INVALID_ARG;
	}
	if (!ls->vt.gels_z_i8) {
		MATX_ERROR("%s: operation not supported", __func__);
		return MATX_ERR_NOT_SUPPORTED;
	}
	return ls->vt.gels_z_i8(A, b, x);
}

// ---- SYEV wrapper ----

matx_status_t matx_syev_d_i8(const matx_dense_linsolve_t* ls,
	const matx_dense_d_i8_t A, matx_vec_d_i8_t eigenvalues, matx_dense_d_i8_t* eigenvectors) 
{
	if (!ls || !A || !eigenvalues) {
		MATX_ERROR("%s: invalid argument", __func__);
		return MATX_ERR_INVALID_ARG;
	}
	if (!ls->vt.syev_d_i8) {
		MATX_ERROR("%s: operation not supported", __func__);
		return MATX_ERR_NOT_SUPPORTED;
	}
	return ls->vt.syev_d_i8(A, eigenvalues, eigenvectors);
}

// ---- GESVD wrappers ----

matx_status_t matx_gesvd_d_i8(const matx_dense_linsolve_t* ls,
	const matx_dense_d_i8_t A, matx_vec_d_i8_t S, matx_dense_d_i8_t* U, matx_dense_d_i8_t* Vt) 
{
	if (!ls || !A || !S) {
		MATX_ERROR("%s: invalid argument", __func__);
		return MATX_ERR_INVALID_ARG;
	}
	if (!ls->vt.gesvd_d_i8) {
		MATX_ERROR("%s: operation not supported", __func__);
		return MATX_ERR_NOT_SUPPORTED;
	}
	return ls->vt.gesvd_d_i8(A, S, U, Vt);
}

matx_status_t matx_gesvd_z_i8(const matx_dense_linsolve_t* ls,
	const matx_dense_z_i8_t A, matx_vec_d_i8_t S, matx_dense_z_i8_t* U, matx_dense_z_i8_t* Vt) 
{
	if (!ls || !A || !S) {
		MATX_ERROR("%s: invalid argument", __func__);
		return MATX_ERR_INVALID_ARG;
	}
	if (!ls->vt.gesvd_z_i8) {
		MATX_ERROR("%s: operation not supported", __func__);
		return MATX_ERR_NOT_SUPPORTED;
	}
	return ls->vt.gesvd_z_i8(A, S, U, Vt);
}

