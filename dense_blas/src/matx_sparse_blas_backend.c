#include "matx/matx_compute.h"

#include <string.h>

// Forward decls
matx_sparse_backend_t matx_sparse_make_reference_grb(void);
matx_sparse_backend_t matx_sparse_make_reference_mkl(void);

const char* matx_sparse_backend_name(matx_sparse_backend_kind_t k) {
	switch (k) {
	case MATX_BLAS_BACKEND_REFERENCE: return "REFERENCE";
	case MATX_BLAS_BACKEND_OPENBLAS: return "OPENBLAS";
	case MATX_BLAS_BACKEND_BLIS: return "BLIS";
	default: return "UNKNOWN";
	}
}

static matx_sparse_backend_t choose_default_backend(void) {
	// Build-time selection (simple & portable). Can be extended to runtime CPUID switching later.
	// If user wants strict control: set -DMATX_BLAS_BACKEND=OPENBLAS/BLIS/REFERENCE
#if defined(MATX_BLAS_BACKEND_REFERENCE_ONLY)
	return matx_blas_make_reference();
#else
  // If no external backend is wired in, fall back to reference.
	return matx_sparse_make_reference_grb();
#endif
}

matx_sparse_backend_t matx_sparse_default(void) {
	return choose_default_backend();
}

matx_status_t matx_spmv_coo_z_i8(const matx_sparse_backend_t* backend,
	matx_complex_d_i8 alpha,
	matx_coo_z_i8_t* A,
	matx_vec_z_i8_t* x,
	matx_complex_d_i8 beta,
	matx_vec_z_i8_t* y)
{
	return backend->vt.spmv_z_i8(alpha, A, x, beta, y);
	return MATX_OK;
}

matx_status_t matx_spmm_coo_z_i8(const matx_sparse_backend_t* backend,
	matx_complex_d_i8 alpha,
	const matx_coo_z_i8_t* A,
	const matx_dense_z_i8_t* B,
	matx_complex_d_i8 beta,
	matx_dense_z_i8_t* C)
{
	return backend->vt.spmm_z_i8(alpha, A, B, beta, C);
}

matx_status_t matx_spmv_coo_d_i8(const matx_sparse_backend_t* backend,
	matx_double alpha,
	matx_coo_d_i8_t* A,
	matx_vec_d_i8_t* x,
	matx_double beta,
	matx_vec_d_i8_t* y)
{
	return backend->vt.spmv_d_i8(alpha, A, x, beta, y);
}

matx_status_t matx_spmm_coo_d_i8(const matx_sparse_backend_t* backend,
	matx_double alpha,
	matx_coo_d_i8_t* A,
	matx_dense_d_i8_t* B,
	matx_double beta,
	matx_dense_d_i8_t* C)
{
	return backend->vt.spmm_d_i8(alpha, A, B, beta, C);
}