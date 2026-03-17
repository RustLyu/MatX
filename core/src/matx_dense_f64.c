#include "matx/matx_types.h"
#include "matx/matx_func.h"

matx_status_t matx_dense_f64_create(matx_dense_f64_t* out,
	matx_int64_t rows,
	matx_int64_t cols,
	matx_layout_t layout,
	const matx_alloc_t* alloc) {
	if (!out || rows == 0 || cols == 0) return MATX_ERR_INVALID_ARG;
	if (layout != MATX_COL_MAJOR && layout != MATX_ROW_MAJOR) return MATX_ERR_INVALID_ARG;
	if (!alloc) return MATX_ERR_INVALID_ARG;

	memset(out, 0, sizeof(*out));
	out->rows = rows;
	out->cols = cols;
	out->layout = layout;
	out->stride = (layout == MATX_COL_MAJOR) ? rows : cols;
	out->flags = 1u;  // owns data

	const size_t n = rows * cols;
	out->data = (matx_double*)matx_malloc(alloc, n * sizeof(matx_double));
	if (!out->data) {
		memset(out, 0, sizeof(*out));
		return MATX_ERR_OUT_OF_MEMORY;
	}
	return MATX_OK;
}

matx_status_t matx_dense_f64_wrap(matx_dense_f64_t* out,
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

	out->rows = rows;
	out->cols = cols;
	out->stride = stride;
	out->layout = layout;
	out->data = data;
	out->flags = 0u;  // does not own
	return MATX_OK;
}

void matx_dense_f64_destroy(matx_dense_f64_t* m, const matx_alloc_t* alloc) {
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

	memset(m, 0, sizeof(*m));
}

