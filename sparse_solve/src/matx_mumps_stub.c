#include "matx/matx_sparse_solve.h"

struct matx_factor_sparse_f64_t { int unused; };
struct matx_factor_sparse_c64_t { int unused; };

static matx_status_t mumps_factor_csc_f64(matx_coo_f64_t A, matx_factor_sparse_f64_t** out_F)
{
	(void)A; (void)out_F;
	return MATX_ERR_NOT_SUPPORTED;
}

static matx_status_t mumps_solve_csc_f64(matx_factor_sparse_f64_t* F, const matx_double* b, matx_double* x)
{
	(void)F; (void)b; (void)x;
	return MATX_ERR_NOT_SUPPORTED;
}

static void mumps_factor_csc_f64_destroy(matx_factor_sparse_f64_t* F)
{
	free(F);
}

static matx_status_t mumps_factor_csc_c64(matx_coo_c64_t A, matx_factor_sparse_c64_t** out_F)
{
	(void)A; (void)out_F;
	return MATX_ERR_NOT_SUPPORTED;
}

static matx_status_t mumps_solve_csc_c64(matx_factor_sparse_c64_t* F, const matx_vec_c64_t b, matx_vec_c64_t x)
{
	(void)F; (void)b; (void)x;
	return MATX_ERR_NOT_SUPPORTED;
}

static void mumps_factor_csc_c64_destroy(matx_factor_sparse_c64_t* F)
{
	free(F);
}

matx_sparse_linsolve_t matx_linsolve_make_mumps(void)
{
	matx_sparse_linsolve_t ls = {
		.kind = MATX_LINSOLVE_BACKEND_MUMPS,
		.vt = {
			.factor_csc_f64 = &mumps_factor_csc_f64,
			.solve_csc_f64 = &mumps_solve_csc_f64,
			.factor_csc_f64_destroy = &mumps_factor_csc_f64_destroy,
			.factor_csc_c64 = &mumps_factor_csc_c64,
			.solve_csc_c64 = &mumps_solve_csc_c64,
			.factor_csc_c64_destroy = &mumps_factor_csc_c64_destroy
		}
	};
	return ls;
}
