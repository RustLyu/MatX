#include "matx/matx_dense_solve.h"
#include "matx/matx_log.h"
#include "matx/matx_types_internal.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#if MATX_ENABLE_OPENBLAS
    #include "openblas/cblas.h"
    #include "openblas/lapacke.h"
#elif MATX_ENABLE_LIBFLAME
	//#include "FLAME.h"
	#include "lapacke.h"
#endif

// Dense factorization
struct matx_factor_dense_f64_t {
	matx_int64_t n;
	matx_int64_t lda;
	matx_double* lu;
	matx_int64_t* piv;
	matx_layout_t layout;
	int uplo;
};

struct matx_factor_dense_c64_t {
	matx_int64_t n;
	matx_int64_t lda;
	matx_double* lu;
	matx_int64_t* piv;
	matx_layout_t layout;
	int uplo;
};

static void ss_factor_dense_f64_destroy(matx_factor_dense_f64_t* F) {
	if (!F) return;
	free(F->lu);
	free(F->piv);
	free(F);
}

static void ss_factor_dense_c64_destroy(matx_factor_dense_c64_t* F) {
	if (!F) return;
	free(F->lu);
	free(F->piv);
	free(F);
}

// ---- Stride-aware packing helpers ----
// Pack a strided dense matrix into a contiguous buffer with lda = min dimension.
// Returns lda for the packed buffer.

static matx_int64_t ss_packed_lda(matx_layout_t layout, matx_int64_t nrows, matx_int64_t ncols) {
	return (layout == MATX_COL_MAJOR) ? nrows : ncols;
}

static void ss_pack_f64(matx_layout_t layout, matx_int64_t nrows, matx_int64_t ncols,
	matx_int64_t src_stride, const matx_double* src, matx_double* dst)
{
	matx_int64_t lda = ss_packed_lda(layout, nrows, ncols);
	if (src_stride == lda) {
		memcpy(dst, src, (size_t)nrows * (size_t)ncols * sizeof(matx_double));
	} else {
		for (matx_int64_t j = 0; j < ncols; ++j)
			for (matx_int64_t i = 0; i < nrows; ++i) {
				matx_int64_t si = (layout == MATX_COL_MAJOR) ? i + j * src_stride : j + i * src_stride;
				matx_int64_t di = (layout == MATX_COL_MAJOR) ? i + j * lda : j + i * lda;
				dst[di] = src[si];
			}
	}
}

static void ss_pack_c64(matx_layout_t layout, matx_int64_t nrows, matx_int64_t ncols,
	matx_int64_t src_stride, const matx_complex_f64_t* src, matx_complex_f64_t* dst)
{
	matx_int64_t lda = ss_packed_lda(layout, nrows, ncols);
	if (src_stride == lda) {
		memcpy(dst, src, (size_t)nrows * (size_t)ncols * sizeof(matx_complex_f64_t));
	} else {
		for (matx_int64_t j = 0; j < ncols; ++j)
			for (matx_int64_t i = 0; i < nrows; ++i) {
				matx_int64_t si = (layout == MATX_COL_MAJOR) ? i + j * src_stride : j + i * src_stride;
				matx_int64_t di = (layout == MATX_COL_MAJOR) ? i + j * lda : j + i * lda;
				dst[di] = src[si];
			}
	}
}

static int ss_layout_to_lapack(matx_layout_t layout) {
	return (layout == MATX_COL_MAJOR) ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
}

// ---- Dense real LU ----

static matx_status_t ss_factor_dense_f64(const matx_dense_f64_t A,
	matx_factor_dense_f64_t** out_F) {
	if (!A || !out_F)
		return MATX_ERR_INVALID_ARG;

	if (A->nrows != A->ncols)
		return MATX_ERR_INVALID_ARG;

	if (*out_F != NULL)
		ss_factor_dense_f64_destroy(*out_F);

	const matx_int64_t n = A->nrows;
	matx_int64_t lda = ss_packed_lda(A->layout, n, n);

	matx_factor_dense_f64_t* F = (matx_factor_dense_f64_t*)malloc(sizeof(*F));
	if (!F) return MATX_ERR_OUT_OF_MEMORY;
	memset(F, 0, sizeof(*F));
	F->n = n;
	F->lda = lda;
	F->layout = A->layout;

	F->lu = (matx_double*)malloc((size_t)n * (size_t)n * sizeof(matx_double));
	F->piv = (matx_int64_t*)malloc(n * sizeof(matx_int64_t));
	if (!F->lu || !F->piv) {
		free(F->lu); free(F->piv); free(F);
		return MATX_ERR_OUT_OF_MEMORY;
	}

	ss_pack_f64(A->layout, n, n, A->stride, A->data, F->lu);

	matx_int64_t info = LAPACKE_dgetrf(ss_layout_to_lapack(A->layout),
		n, n, F->lu, lda, F->piv);
	if (info != 0) {
		MATX_ERROR("LAPACKE_dgetrf error:%d", info);
		free(F->lu); free(F->piv); free(F);
		return MATX_ERR_INTERNAL;
	}

	*out_F = F;
	return MATX_OK;
}

static matx_status_t ss_solve_dense_f64(const matx_factor_dense_f64_t* F,
	const matx_double* b, matx_double* x)
{
	if (!F || !b || !x) return MATX_ERR_INVALID_ARG;
	memcpy(x, b, F->n * sizeof(matx_double));

	matx_int64_t ldb = (F->layout == MATX_COL_MAJOR) ? F->n : 1;
	matx_int64_t info = LAPACKE_dgetrs(ss_layout_to_lapack(F->layout),
		'N', F->n, 1, F->lu, F->lda, F->piv, x, ldb);
	if (info != 0) {
		MATX_ERROR("LAPACKE_dgetrs error:%d", info);
		return MATX_ERR_INTERNAL;
	}
	return MATX_OK;
}

// ---- Dense complex LU ----

static matx_status_t ss_factor_dense_c64(const matx_dense_c64_t A,
	matx_factor_dense_c64_t** out_F)
{
#if !(defined(MATX_HAVE_OPENBLAS) || defined(MATX_HAVE_BLIS))
	return MATX_ERR_NOT_SUPPORTED;
#else
	if (!A || !out_F) return MATX_ERR_INVALID_ARG;
	if (A->nrows != A->ncols) return MATX_ERR_INVALID_ARG;

	if (*out_F != NULL) ss_factor_dense_c64_destroy(*out_F);

	const matx_int64_t n = A->nrows;
	matx_int64_t lda = ss_packed_lda(A->layout, n, n);

	matx_factor_dense_c64_t* F = (matx_factor_dense_c64_t*)malloc(sizeof(*F));
	if (!F) return MATX_ERR_OUT_OF_MEMORY;
	memset(F, 0, sizeof(*F));
	F->n = n;
	F->lda = lda;
	F->layout = A->layout;

	F->lu = (matx_double*)malloc((size_t)n * (size_t)n * sizeof(matx_complex_f64_t));
	F->piv = (matx_int64_t*)malloc(n * sizeof(matx_int64_t));
	if (!F->lu || !F->piv) {
		free(F->lu); free(F->piv); free(F);
		return MATX_ERR_OUT_OF_MEMORY;
	}

	ss_pack_c64(A->layout, n, n, A->stride, A->data, (matx_complex_f64_t*)F->lu);

	matx_int64_t info = LAPACKE_zgetrf(ss_layout_to_lapack(A->layout),
		n, n, (lapack_complex_double*)F->lu, lda, F->piv);
	if (info != 0) {
		MATX_ERROR("LAPACKE_zgetrf error:%d", info);
		free(F->lu); free(F->piv); free(F);
		return MATX_ERR_INTERNAL;
	}

	*out_F = F;
	return MATX_OK;
#endif
}

static matx_status_t ss_solve_dense_c64(const matx_factor_dense_c64_t* F,
	const matx_vec_c64_t b, matx_vec_c64_t x)
{
	if (!F || !b || !x) return MATX_ERR_INVALID_ARG;
	memcpy(x->data, b->data, F->n * sizeof(matx_complex_f64_t));

	matx_int64_t ldb = (F->layout == MATX_COL_MAJOR) ? F->n : 1;
	matx_int64_t info = LAPACKE_zgetrs(ss_layout_to_lapack(F->layout),
		'N', F->n, 1, (lapack_complex_double*)F->lu, F->lda, F->piv,
		(lapack_complex_double*)x->data, ldb);
	if (info != 0) {
		MATX_ERROR("LAPACKE_zgetrs error:%d", info);
		return MATX_ERR_INTERNAL;
	}
	return MATX_OK;
}

// ---- Cholesky ----

static matx_status_t ss_potrf_f64(const matx_dense_f64_t A, int uplo, matx_factor_dense_f64_t** out_F) {
	if (!A || !out_F) return MATX_ERR_INVALID_ARG;
	if (A->nrows != A->ncols) return MATX_ERR_INVALID_ARG;

	if (*out_F) ss_factor_dense_f64_destroy(*out_F);

	const matx_int64_t n = A->nrows;
	matx_int64_t lda = ss_packed_lda(A->layout, n, n);

	matx_factor_dense_f64_t* F = (matx_factor_dense_f64_t*)malloc(sizeof(*F));
	if (!F) return MATX_ERR_OUT_OF_MEMORY;
	memset(F, 0, sizeof(*F));
	F->n = n;
	F->lda = lda;
	F->layout = A->layout;
	F->uplo = uplo;

	F->lu = (matx_double*)malloc((size_t)n * (size_t)n * sizeof(matx_double));
	if (!F->lu) { free(F); return MATX_ERR_OUT_OF_MEMORY; }

	ss_pack_f64(A->layout, n, n, A->stride, A->data, F->lu);

	matx_int64_t info = LAPACKE_dpotrf(ss_layout_to_lapack(A->layout),
		uplo, n, F->lu, lda);
	if (info != 0) {
		MATX_ERROR("LAPACKE_dpotrf error: %d", info);
		free(F->lu); free(F);
		return MATX_ERR_INTERNAL;
	}
	*out_F = F;
	return MATX_OK;
}

static matx_status_t ss_potrs_f64(const matx_factor_dense_f64_t* F, const matx_double* b, matx_double* x)
{
	if (!F || !b || !x) return MATX_ERR_INVALID_ARG;
	memcpy(x, b, F->n * sizeof(matx_double));

	matx_int64_t ldb = (F->layout == MATX_COL_MAJOR) ? F->n : 1;
	matx_int64_t info = LAPACKE_dpotrs(ss_layout_to_lapack(F->layout),
		F->uplo, F->n, 1, F->lu, F->lda, x, ldb);
	if (info != 0) {
		MATX_ERROR("LAPACKE_dpotrs error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	return MATX_OK;
}

static matx_status_t ss_potrf_c64(const matx_dense_c64_t A, int uplo, matx_factor_dense_c64_t** out_F) {
	if (!A || !out_F) return MATX_ERR_INVALID_ARG;
	if (A->nrows != A->ncols) return MATX_ERR_INVALID_ARG;

	if (*out_F) ss_factor_dense_c64_destroy(*out_F);

	const matx_int64_t n = A->nrows;
	matx_int64_t lda = ss_packed_lda(A->layout, n, n);

	matx_factor_dense_c64_t* F = (matx_factor_dense_c64_t*)malloc(sizeof(*F));
	if (!F) return MATX_ERR_OUT_OF_MEMORY;
	memset(F, 0, sizeof(*F));
	F->n = n;
	F->lda = lda;
	F->layout = A->layout;
	F->uplo = uplo;

	F->lu = (matx_double*)malloc((size_t)n * (size_t)n * sizeof(matx_complex_f64_t));
	if (!F->lu) { free(F); return MATX_ERR_OUT_OF_MEMORY; }

	ss_pack_c64(A->layout, n, n, A->stride, A->data, (matx_complex_f64_t*)F->lu);

	matx_int64_t info = LAPACKE_zpotrf(ss_layout_to_lapack(A->layout),
		uplo, n, (lapack_complex_double*)F->lu, lda);
	if (info != 0) {
		MATX_ERROR("LAPACKE_zpotrf error: %d", info);
		free(F->lu); free(F);
		return MATX_ERR_INTERNAL;
	}
	*out_F = F;
	return MATX_OK;
}

static matx_status_t ss_potrs_c64(const matx_factor_dense_c64_t* F, const matx_vec_c64_t b, matx_vec_c64_t x) {
	if (!F || !b || !x) return MATX_ERR_INVALID_ARG;
	memcpy(x->data, b->data, F->n * sizeof(matx_complex_f64_t));

	matx_int64_t ldb = (F->layout == MATX_COL_MAJOR) ? F->n : 1;
	matx_int64_t info = LAPACKE_zpotrs(ss_layout_to_lapack(F->layout),
		F->uplo, F->n, 1, (lapack_complex_double*)F->lu, F->lda,
		(lapack_complex_double*)x->data, ldb);
	if (info != 0) {
		MATX_ERROR("LAPACKE_zpotrs error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	return MATX_OK;
}

// ---- GELS ----

static matx_status_t ss_gels_f64(const matx_dense_f64_t A, const matx_double* b, matx_double* x) {
	if (!A || !b || !x) return MATX_ERR_INVALID_ARG;
	const matx_int64_t m = A->nrows, n = A->ncols;
	const matx_int64_t lda = ss_packed_lda(A->layout, m, n);
	const matx_int64_t blen = (m > n) ? m : n;
	const matx_int64_t ldb = (A->layout == MATX_COL_MAJOR) ? blen : 1;

	matx_double* Acopy = (matx_double*)malloc((size_t)m * (size_t)n * sizeof(matx_double));
	matx_double* bcopy = (matx_double*)calloc(blen, sizeof(matx_double));
	if (!Acopy || !bcopy) { free(Acopy); free(bcopy); return MATX_ERR_OUT_OF_MEMORY; }

	ss_pack_f64(A->layout, m, n, A->stride, A->data, Acopy);
	memcpy(bcopy, b, m * sizeof(matx_double));

	matx_int64_t info = LAPACKE_dgels(ss_layout_to_lapack(A->layout),
		'N', m, n, 1, Acopy, lda, bcopy, ldb);
	if (info != 0) {
		MATX_ERROR("LAPACKE_dgels error: %d", info);
		free(Acopy); free(bcopy);
		return MATX_ERR_INTERNAL;
	}
	memcpy(x, bcopy, n * sizeof(matx_double));
	free(Acopy); free(bcopy);
	return MATX_OK;
}

static matx_status_t ss_gels_c64(const matx_dense_c64_t A, const matx_vec_c64_t b, matx_vec_c64_t x) {
	if (!A || !b || !x) return MATX_ERR_INVALID_ARG;
	const matx_int64_t m = A->nrows, n = A->ncols;
	const matx_int64_t lda = ss_packed_lda(A->layout, m, n);
	const matx_int64_t blen = (m > n) ? m : n;
	const matx_int64_t ldb = (A->layout == MATX_COL_MAJOR) ? blen : 1;

	matx_complex_f64_t* Acopy = (matx_complex_f64_t*)malloc((size_t)m * (size_t)n * sizeof(matx_complex_f64_t));
	matx_complex_f64_t* bcopy = (matx_complex_f64_t*)calloc(blen, sizeof(matx_complex_f64_t));
	if (!Acopy || !bcopy) { free(Acopy); free(bcopy); return MATX_ERR_OUT_OF_MEMORY; }

	ss_pack_c64(A->layout, m, n, A->stride, A->data, Acopy);
	memcpy(bcopy, b->data, m * sizeof(matx_complex_f64_t));

	matx_int64_t info = LAPACKE_zgels(ss_layout_to_lapack(A->layout),
		'N', m, n, 1, (lapack_complex_double*)Acopy, lda,
		(lapack_complex_double*)bcopy, ldb);
	if (info != 0) {
		MATX_ERROR("LAPACKE_zgels error: %d", info);
		free(Acopy); free(bcopy);
		return MATX_ERR_INTERNAL;
	}
	memcpy(x->data, bcopy, n * sizeof(matx_complex_f64_t));
	free(Acopy); free(bcopy);
	return MATX_OK;
}

// ---- SYEV ----

static matx_status_t ss_syev_f64(const matx_dense_f64_t A, matx_vec_f64_t eigenvalues, matx_dense_f64_t* eigenvectors) {
	if (!A || !eigenvalues) return MATX_ERR_INVALID_ARG;
	if (A->nrows != A->ncols) return MATX_ERR_INVALID_ARG;
	const matx_int64_t n = A->nrows;
	if (eigenvalues->n != n) return MATX_ERR_INVALID_ARG;

	const char jobz = eigenvectors ? 'V' : 'N';
	matx_int64_t lda = ss_packed_lda(A->layout, n, n);

	matx_double* Acopy = (matx_double*)malloc((size_t)n * (size_t)n * sizeof(matx_double));
	if (!Acopy) return MATX_ERR_OUT_OF_MEMORY;

	ss_pack_f64(A->layout, n, n, A->stride, A->data, Acopy);

	matx_int64_t info = LAPACKE_dsyev(ss_layout_to_lapack(A->layout),
		jobz, 'L', n, Acopy, lda, eigenvalues->data);
	if (info != 0) {
		MATX_ERROR("LAPACKE_dsyev error: %d", info);
		free(Acopy);
		return MATX_ERR_INTERNAL;
	}
	if (eigenvectors) {
		matx_dense_f64_opaque_t* ev = (matx_dense_f64_opaque_t*)malloc(sizeof(matx_dense_f64_opaque_t));
		if (!ev) { free(Acopy); return MATX_ERR_OUT_OF_MEMORY; }
		memset(ev, 0, sizeof(*ev));
		ev->nrows = n; ev->ncols = n;
		ev->layout = A->layout;
		ev->stride = lda;
		ev->flags = 1u;
		ev->data = Acopy;
		if (*eigenvectors) {
			if ((*eigenvectors)->flags & 1u) free((*eigenvectors)->data);
			free(*eigenvectors);
		}
		*eigenvectors = ev;
	} else {
		free(Acopy);
	}
	return MATX_OK;
}

// ---- GESVD ----

static matx_status_t ss_gesvd_f64(const matx_dense_f64_t A, matx_vec_f64_t S, matx_dense_f64_t* U, matx_dense_f64_t* Vt) {
	if (!A || !S) return MATX_ERR_INVALID_ARG;
	const matx_int64_t m = A->nrows, n = A->ncols;
	const matx_int64_t k = (m < n) ? m : n;
	if (S->n != k) return MATX_ERR_INVALID_ARG;

	matx_int64_t lda = ss_packed_lda(A->layout, m, n);

	matx_double* Acopy = (matx_double*)malloc((size_t)m * (size_t)n * sizeof(matx_double));
	matx_double* u_data = U ? (matx_double*)malloc((size_t)m * (size_t)m * sizeof(matx_double)) : NULL;
	matx_double* vt_data = Vt ? (matx_double*)malloc((size_t)n * (size_t)n * sizeof(matx_double)) : NULL;
	matx_double* superb = (matx_double*)malloc(k * sizeof(matx_double));
	if (!Acopy || !superb || (U && !u_data) || (Vt && !vt_data)) {
		free(Acopy); free(u_data); free(vt_data); free(superb);
		return MATX_ERR_OUT_OF_MEMORY;
	}

	ss_pack_f64(A->layout, m, n, A->stride, A->data, Acopy);

	const char jobu = U ? 'A' : 'N';
	const char jobvt = Vt ? 'A' : 'N';
	matx_int64_t ldu = m, ldvt = n;
	matx_int64_t info = LAPACKE_dgesvd(ss_layout_to_lapack(A->layout),
		jobu, jobvt, m, n, Acopy, lda, S->data,
		u_data, ldu, vt_data, ldvt, superb);
	free(Acopy); free(superb);
	if (info != 0) {
		MATX_ERROR("LAPACKE_dgesvd error: %d", info);
		free(u_data); free(vt_data);
		return MATX_ERR_INTERNAL;
	}
	if (U) {
		matx_int64_t u_lda = ss_packed_lda(A->layout, m, m);
		matx_dense_f64_opaque_t* ev = (matx_dense_f64_opaque_t*)malloc(sizeof(matx_dense_f64_opaque_t));
		if (!ev) { free(u_data); free(vt_data); return MATX_ERR_OUT_OF_MEMORY; }
		memset(ev, 0, sizeof(*ev));
		ev->nrows = m; ev->ncols = m; ev->layout = A->layout; ev->stride = u_lda; ev->flags = 1u; ev->data = u_data;
		if (*U) { if ((*U)->flags & 1u) free((*U)->data); free(*U); }
		*U = ev;
	}
	if (Vt) {
		matx_int64_t vt_lda = ss_packed_lda(A->layout, n, n);
		matx_dense_f64_opaque_t* ev = (matx_dense_f64_opaque_t*)malloc(sizeof(matx_dense_f64_opaque_t));
		if (!ev) { free(vt_data); return MATX_ERR_OUT_OF_MEMORY; }
		memset(ev, 0, sizeof(*ev));
		ev->nrows = n; ev->ncols = n; ev->layout = A->layout; ev->stride = vt_lda; ev->flags = 1u; ev->data = vt_data;
		if (*Vt) { if ((*Vt)->flags & 1u) free((*Vt)->data); free(*Vt); }
		*Vt = ev;
	}
	return MATX_OK;
}

static matx_status_t ss_gesvd_c64(const matx_dense_c64_t A, matx_vec_f64_t S, matx_dense_c64_t* U, matx_dense_c64_t* Vt) {
	if (!A || !S) return MATX_ERR_INVALID_ARG;
	const matx_int64_t m = A->nrows, n = A->ncols;
	const matx_int64_t k = (m < n) ? m : n;
	if (S->n != k) return MATX_ERR_INVALID_ARG;

	matx_int64_t lda = ss_packed_lda(A->layout, m, n);

	matx_complex_f64_t* Acopy = (matx_complex_f64_t*)malloc((size_t)m * (size_t)n * sizeof(matx_complex_f64_t));
	matx_complex_f64_t* u_data = U ? (matx_complex_f64_t*)malloc((size_t)m * (size_t)m * sizeof(matx_complex_f64_t)) : NULL;
	matx_complex_f64_t* vt_data = Vt ? (matx_complex_f64_t*)malloc((size_t)n * (size_t)n * sizeof(matx_complex_f64_t)) : NULL;
	matx_double* superb = (matx_double*)malloc(k * sizeof(matx_double));
	if (!Acopy || !superb || (U && !u_data) || (Vt && !vt_data)) {
		free(Acopy); free(u_data); free(vt_data); free(superb);
		return MATX_ERR_OUT_OF_MEMORY;
	}

	ss_pack_c64(A->layout, m, n, A->stride, A->data, Acopy);

	const char jobu = U ? 'A' : 'N';
	const char jobvt = Vt ? 'A' : 'N';
	matx_int64_t ldu = m, ldvt = n;
	matx_int64_t info = LAPACKE_zgesvd(ss_layout_to_lapack(A->layout),
		jobu, jobvt, m, n, (lapack_complex_double*)Acopy, lda, S->data,
		(lapack_complex_double*)u_data, ldu,
		(lapack_complex_double*)vt_data, ldvt, superb);
	free(Acopy); free(superb);
	if (info != 0) {
		MATX_ERROR("LAPACKE_zgesvd error: %d", info);
		free(u_data); free(vt_data);
		return MATX_ERR_INTERNAL;
	}
	if (U) {
		matx_int64_t u_lda = ss_packed_lda(A->layout, m, m);
		matx_dense_c64_opaque_t* ev = (matx_dense_c64_opaque_t*)malloc(sizeof(matx_dense_c64_opaque_t));
		if (!ev) { free(u_data); free(vt_data); return MATX_ERR_OUT_OF_MEMORY; }
		memset(ev, 0, sizeof(*ev));
		ev->nrows = m; ev->ncols = m; ev->layout = A->layout; ev->stride = u_lda; ev->flags = 1u; ev->data = u_data;
		if (*U) { if ((*U)->flags & 1u) free((*U)->data); free(*U); }
		*U = ev;
	}
	if (Vt) {
		matx_int64_t vt_lda = ss_packed_lda(A->layout, n, n);
		matx_dense_c64_opaque_t* ev = (matx_dense_c64_opaque_t*)malloc(sizeof(matx_dense_c64_opaque_t));
		if (!ev) { free(vt_data); return MATX_ERR_OUT_OF_MEMORY; }
		memset(ev, 0, sizeof(*ev));
		ev->nrows = n; ev->ncols = n; ev->layout = A->layout; ev->stride = vt_lda; ev->flags = 1u; ev->data = vt_data;
		if (*Vt) { if ((*Vt)->flags & 1u) free((*Vt)->data); free(*Vt); }
		*Vt = ev;
	}
	return MATX_OK;
}

matx_dense_linsolve_t matx_dense_linsolve_make_cblas(void) {
	matx_dense_linsolve_t ls = {
		.kind = MATX_LINSOLVE_BACKEND_CBLAS,
		.vt = {
			.factor_dense_f64 = &ss_factor_dense_f64,
			.solve_dense_f64 = &ss_solve_dense_f64,
			.factor_dense_f64_destroy = &ss_factor_dense_f64_destroy,
			.factor_dense_c64 = &ss_factor_dense_c64,
			.solve_dense_c64 = &ss_solve_dense_c64,
			.factor_dense_c64_destroy = &ss_factor_dense_c64_destroy,
			.potrf_f64 = &ss_potrf_f64,
			.potrs_f64 = &ss_potrs_f64,
			.potrf_c64 = &ss_potrf_c64,
			.potrs_c64 = &ss_potrs_c64,
			.gels_f64 = &ss_gels_f64,
			.gels_c64 = &ss_gels_c64,
			.syev_f64 = &ss_syev_f64,
			.gesvd_f64 = &ss_gesvd_f64,
			.gesvd_c64 = &ss_gesvd_c64,
		}
	};

	return ls;
}