#include "matx/matx_types.h"
#include "matx/matx_func.h"
#include "matx/matx_types_internal.h"

#include <string.h>
#include <stdio.h>
#include <math.h>

matx_status_t matx_dense_d_i8_create(
	const matx_alloc_t* alloc,
	matx_dense_d_i8_t* out,
	matx_layout_t layout,
	matx_int64_t rows,
	matx_int64_t cols,
	matx_double* data) {
	if (!out || rows == 0 || cols == 0)
		return MATX_ERR_INVALID_ARG;
	if (layout != MATX_COL_MAJOR && layout != MATX_ROW_MAJOR)
		return MATX_ERR_INVALID_ARG;
	if (!alloc)
		return MATX_ERR_INVALID_ARG;

	matx_dense_d_i8_opaque_t* out_value = matx_malloc(alloc, sizeof(matx_dense_d_i8_opaque_t));
	if (!out_value) return MATX_ERR_OUT_OF_MEMORY;
	memset(out_value, 0, sizeof(matx_dense_d_i8_opaque_t));
	out_value->nrows = rows;
	out_value->ncols = cols;
	out_value->layout = layout;
	out_value->stride = (layout == MATX_COL_MAJOR) ? rows : cols;
	out_value->flags = 1u;

	const size_t n = rows * cols;
	out_value->data = (matx_double*)matx_malloc(alloc, n * sizeof(matx_double));
	if (!out_value->data) {
		matx_free(alloc, out_value);
		return MATX_ERR_OUT_OF_MEMORY;
	}
	if (data != NULL)
	{
		memcpy(out_value->data, data, sizeof(matx_double) * n);
	}
	if (*out != NULL)
	{
		matx_dense_d_i8_destroy(alloc, *out);
	}
	*out = out_value;
	return MATX_OK;
}

matx_status_t matx_dense_d_i8_dup(const matx_alloc_t* alloc, const matx_dense_d_i8_t in, matx_dense_d_i8_t* out)
{
	return matx_dense_d_i8_create(alloc, out, in->layout, in->nrows, in->ncols, in->data);
}

matx_status_t matx_dense_d_i8_wrap(const matx_alloc_t* alloc,
	matx_dense_d_i8_t* out,
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
	matx_dense_d_i8_opaque_t* out_value = matx_malloc(alloc, sizeof(matx_dense_d_i8_opaque_t));
	if (!out_value) return MATX_ERR_OUT_OF_MEMORY;
	memset(out_value, 0, sizeof(matx_dense_d_i8_opaque_t));

	out_value->nrows = rows;
	out_value->ncols = cols;
	out_value->stride = stride;
	out_value->layout = layout;
	out_value->data = data;
	out_value->flags = 0u;

	if (*out != NULL)
	{
		matx_dense_d_i8_destroy(alloc, *out);
	}

	*out = out_value;
	return MATX_OK;
}

void matx_dense_d_i8_destroy(const matx_alloc_t* alloc, matx_dense_d_i8_t m) {
	if (!m)
		return;
	if ((m->flags & 1u) != 0u && m->data) {
		if (alloc)
		{
			matx_free(alloc, m->data);
			m->data = NULL;
		}
	}
	else if (m->flags == 0u)
	{
		m->data = NULL;
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
}

matx_status_t matx_dense_d_i8_fill(matx_dense_d_i8_t m, matx_double val) {
	if (!m || !m->data) return MATX_ERR_INVALID_ARG;
	const matx_int64_t total = m->nrows * m->ncols;
	for (matx_int64_t i = 0; i < total; ++i)
		m->data[i] = val;
	return MATX_OK;
}

matx_status_t matx_dense_d_i8_zeros(matx_dense_d_i8_t m) {
	if (!m || !m->data) return MATX_ERR_INVALID_ARG;
	memset(m->data, 0, (size_t)(m->nrows * m->ncols) * sizeof(matx_double));
	return MATX_OK;
}

matx_status_t matx_dense_d_i8_ones(matx_dense_d_i8_t m) {
	return matx_dense_d_i8_fill(m, 1.0);
}

matx_status_t matx_dense_d_i8_trace(const matx_dense_d_i8_t A, matx_double* out) {
	if (!A || !out) return MATX_ERR_INVALID_ARG;
	if (A->nrows != A->ncols) return MATX_ERR_INVALID_ARG;
	matx_double sum = 0.0;
	for (matx_int64_t i = 0; i < A->nrows; ++i) {
		matx_int64_t idx = (A->layout == MATX_COL_MAJOR) ? i + i * A->stride : i * A->stride + i;
		sum += A->data[idx];
	}
	*out = sum;
	return MATX_OK;
}

matx_status_t matx_dense_d_i8_to_z_i8(const matx_alloc_t* alloc, const matx_dense_d_i8_t A, matx_dense_z_i8_t* out) {
	if (!alloc || !A || !out) return MATX_ERR_INVALID_ARG;
	matx_status_t st = matx_dense_z_i8_create(alloc, out, A->layout, A->nrows, A->ncols, NULL);
	if (st != MATX_OK) return st;
	const matx_int64_t total = A->nrows * A->ncols;
	for (matx_int64_t i = 0; i < total; ++i) {
		(*out)->data[i].real = A->data[i];
		(*out)->data[i].imag = 0.0;
	}
	return MATX_OK;
}