#include "matx/matx_types.h"
#include "matx/matx_func.h"
#include "matx/matx_types_internal.h"

#include <string.h>

matx_status_t matx_dense_f64_create(
	const matx_alloc_t* alloc,
	matx_dense_f64_t* out,
	matx_layout_t layout,
	matx_int64_t rows,
	matx_int64_t cols,
	matx_double* data) {
	if (!out || rows == 0 || cols == 0) return MATX_ERR_INVALID_ARG;
	if (layout != MATX_COL_MAJOR && layout != MATX_ROW_MAJOR) return MATX_ERR_INVALID_ARG;
	if (!alloc) return MATX_ERR_INVALID_ARG;
	
	matx_dense_f64_opaque_t* out_value = matx_malloc(alloc, sizeof(matx_dense_f64_opaque_t));
	memset(out_value, 0, sizeof(matx_dense_f64_opaque_t));
	out_value->nrows = rows;
	out_value->ncols = cols;
	out_value->layout = layout;
	out_value->stride = (layout == MATX_COL_MAJOR) ? rows : cols;
	out_value->flags = 1u;  // owns data

	const size_t n = rows * cols;
	out_value->data = (matx_double*)matx_malloc(alloc, n * sizeof(matx_double));
	if (!out_value->data) {
		memset(out_value, 0, sizeof(*out_value));
		return MATX_ERR_OUT_OF_MEMORY;
	}
	if (data != NULL)
	{
		memcpy(out_value->data, data, sizeof(matx_double) * out_value->nrows * out_value->ncols);
	}
	if (*out != NULL)
	{
		matx_dense_f64_destroy(alloc, *out);
	}
	*out = out_value;
	return MATX_OK;
}

matx_status_t matx_dense_f64_dup(const matx_alloc_t* alloc, const matx_dense_f64_t in, matx_dense_f64_t* out)
{
	return matx_dense_f64_create(alloc, out, in->layout, in->nrows, in->ncols, in->data);
}

matx_status_t matx_dense_f64_wrap(const matx_alloc_t* alloc, 
	matx_dense_f64_t* out,
	matx_int64_t rows,
	matx_int64_t cols,
	matx_int64_t stride,
	matx_layout_t layout,
	matx_double* data) {
	if (!out || !data || rows == 0 || cols == 0) return MATX_ERR_INVALID_ARG;
	if (layout != MATX_COL_MAJOR && layout != MATX_ROW_MAJOR) return MATX_ERR_INVALID_ARG;
	if (layout == MATX_COL_MAJOR) {
		if (stride < rows) return MATX_ERR_INVALID_ARG;
	}
	else {
		if (stride < cols) return MATX_ERR_INVALID_ARG;
	}
	matx_dense_f64_opaque_t* out_value = matx_malloc(alloc, sizeof(matx_dense_f64_opaque_t));
	memset(out_value, 0, sizeof(matx_dense_f64_opaque_t));

	out_value->nrows = rows;
	out_value->ncols = cols;
	out_value->stride = stride;
	out_value->layout = layout;
	out_value->data = data;
	out_value->flags = 0u;  // does not own

	if (*out != NULL)
	{
		matx_dense_f64_destroy(alloc, *out);
	}

	*out = out_value;
	return MATX_OK;
}

void matx_dense_f64_destroy(const matx_alloc_t* alloc, matx_dense_f64_t m) {
	if (!m) return;
	if ((m->flags & 1u) != 0u && m->data) {
		if (alloc) matx_free(alloc, m->data);
	}

	if (m->handle_grb.valid > 0)
	{
		if (m->handle_grb.custom_free_func && m->handle_grb.impl)
		{
			m->handle_grb.custom_free_func(m->handle_grb.impl);
		}
		m->handle_grb.impl = NULL;
		m->handle_grb.valid = -1;
	}
	matx_free(alloc, m);
	memset(m, 0, sizeof(*m));
}

