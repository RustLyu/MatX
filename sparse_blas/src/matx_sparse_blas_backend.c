#include "matx/matx_sparse_compute.h"

// Forward decls
matx_sparse_backend_t matx_sparse_make_reference_grb(void);
matx_sparse_backend_t matx_sparse_make_reference_mkl(void);
matx_sparse_backend_t matx_sparse_make_reference_aocl(void);

const char* matx_sparse_backend_name(matx_sparse_backend_kind_t k) {
	switch (k) {
	case MATX_SPARSE_BACKEND_REFERENCE: return "REFERENCE";
	case MATX_SPARSE_BACKEND_GRAPHBLAS: return "GRAPHBLAS";
	case MATX_SPARSE_BACKEND_MKL: return "MKL";
	default: return "UNKNOWN";
	}
}

static matx_sparse_backend_t choose_default_backend(void) {
	// Build-time selection (simple & portable). Can be extended to runtime CPUID switching later.
	// If user wants strict control: set -DMATX_BLAS_BACKEND=OPENBLAS/BLIS/REFERENCE
#if defined(MATX_SPARSE_BACKEND_REFERENCE)
	return matx_blas_make_reference();
#else
  // If no external backend is wired in, fall back to reference.
	//return matx_sparse_make_reference_grb();
	return matx_sparse_make_reference_aocl();
#endif
}

matx_sparse_backend_t matx_sparse_default(void) {
	return choose_default_backend();
}

matx_status_t matx_spmv_coo_c64(const matx_sparse_backend_t* backend,
	matx_complex_f64 alpha,
	matx_coo_c64_t* A,
	matx_vec_c64_t* x,
	matx_complex_f64 beta,
	matx_vec_c64_t* y)
{
	return backend->vt.spmv_c64(alpha, A, x, beta, y);
}

matx_status_t matx_spmm_coo_c64(const matx_sparse_backend_t* backend,
	matx_complex_f64 alpha,
	const matx_coo_c64_t* A,
	const matx_dense_c64_t* B,
	matx_complex_f64 beta,
	matx_dense_c64_t* C)
{
	return backend->vt.spmm_c64(alpha, A, B, beta, C);
}

matx_status_t matx_spmv_coo_f64(const matx_sparse_backend_t* backend,
	matx_double alpha,
	matx_coo_f64_t* A,
	matx_vec_f64_t* x,
	matx_double beta,
	matx_vec_f64_t* y)
{
	return backend->vt.spmv_f64(alpha, A, x, beta, y);
}

matx_status_t matx_spmm_coo_f64(const matx_sparse_backend_t* backend,
	matx_double alpha,
	matx_coo_f64_t* A,
	matx_dense_f64_t* B,
	matx_double beta,
	matx_dense_f64_t* C)
{
	return backend->vt.spmm_f64(alpha, A, B, beta, C);
}

matx_status_t matx_dsp2md_coo_f64(const matx_sparse_backend_t* backend,
	matx_double alpha,
	matx_coo_f64_t* A,
	matx_coo_f64_t* B,
	matx_double beta,
	matx_dense_f64_t* C)
{
	return backend->vt.dsp2md_f64(alpha, A, B, beta, C);
}