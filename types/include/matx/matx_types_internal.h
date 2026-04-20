#ifndef MATX_TYPES_INTERNAL_H
#define MATX_TYPES_INTERNAL_H

#include "matx/matx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

	// ---- Dense vector (double) ----
	typedef struct matx_vec_f64_opaque_t {
		matx_int64_t n;
		matx_int64_t stride;
		matx_double* data;
		matx_int64_t flags;
		matx_handle_t handle_grb;
	} matx_vec_f64_opaque_t;

	// ---- Dense matrix (double) ----
	typedef struct matx_dense_f64_opaque_t {
		matx_int64_t  nrows;
		matx_int64_t  ncols;
		matx_int64_t  stride;     // leading dimension: if col-major => ld = stride (>= rows); if row-major => ld = stride (>= cols)
		matx_layout_t layout;
		matx_double* data;
		matx_int64_t flags;    // reserved for future (ownership, alignment, etc.)
		matx_handle_t handle_grb;
	} matx_dense_f64_opaque_t;

	// ---- Dense vector (complex) ----
	typedef struct matx_vec_c64_opaque_t {
		matx_int64_t n;
		matx_int64_t stride;
		matx_complex_f64* data;
		matx_int64_t flags;
		matx_handle_t handle_grb;
	} matx_vec_c64_opaque_t;

	// ---- Dense matrix (complex) ----
	typedef struct matx_dense_c64_opaque_t {
		matx_int64_t nrows;
		matx_int64_t ncols;
		matx_int64_t stride;
		matx_layout_t layout;
		matx_complex_f64* data;
		matx_int64_t flags;
		matx_handle_t handle_grb;
	} matx_dense_c64_opaque_t;

	// ---- Sparse CSC (real/complex) ----
	typedef struct matx_csc_f64_opaque_t {
		matx_int64_t nrows;
		matx_int64_t ncols;
		matx_int64_t nnz;
		matx_int64_t* col_ptr;
		matx_int64_t* row_ind;
		matx_double* values;
		matx_int64_t* coo_csc_index_map;
		matx_int64_t struct_update; // coo to csc conversion may involve sorting and duplicate summation, these flags can be used to track whether the structure/values are up to date with the original COO data
		matx_int64_t only_value_update; // if the structure is up to date, but values have been updated, this flag can be set to indicate that only values need to be updated in the CSC representation without redoing the entire COO to CSC conversion
		uint32_t flags;
	} matx_csc_f64_opaque_t;

	typedef struct matx_coo_f64_opaque_t {
		matx_int64_t nrows;
		matx_int64_t ncols;
		matx_int64_t nnz;
		matx_int64_t* rows;
		matx_int64_t* columns;
		matx_double* values;
		matx_int64_t flags;
		matx_handle_t handle_grb;
		matx_handle_t handle_mkl;
		matx_handle_t handle_aocl;
		matx_csc_f64_opaque_t* handle_csc; // for backends that require CSC format, we can lazily convert COO to CSC and store here to avoid repeated conversions
	} matx_coo_f64_opaque_t;

	typedef struct matx_csc_c64_opaque_t {
		matx_int64_t nrows;
		matx_int64_t ncols;
		matx_int64_t nnz;
		const matx_int64_t* col_ptr;
		const matx_int64_t* row_ind;
		const matx_complex_f64* values;
		const matx_int64_t* coo_csc_index_map;
		matx_int64_t struct_update; // coo to csc conversion may involve sorting and duplicate summation, these flags can be used to track whether the structure/values are up to date with the original COO data
		matx_int64_t only_value_update; // if the structure is up to date, but values have been updated, this flag can be set to indicate that only values need to be updated in the CSC representation without redoing the entire COO to CSC conversion
		matx_int64_t flags;
	} matx_csc_c64_opaque_t;

	typedef struct matx_coo_c64_opaque_t {
		matx_int64_t nrows;
		matx_int64_t ncols;
		matx_int64_t nnz;
		matx_int64_t* rows;
		matx_int64_t* columns;
		matx_complex_f64* values;
		matx_int64_t flags;
		matx_handle_t handle_grb;
		matx_handle_t handle_mkl;
		matx_handle_t handle_aocl;
		matx_csc_c64_opaque_t* handle_csc; // for backends that require CSC format, we can lazily convert COO to CSC and store here to avoid repeated conversions
	} matx_coo_c64_opaque_t;
#ifdef __cplusplus
}
#endif

#endif // MATX_TYPES_INTERNAL_H