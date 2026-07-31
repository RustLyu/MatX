#include "matx/matx_types.h"
#include "matx/matx_func.h"
#include "matx/matx_types_internal.h"

#include <string.h>
#include <math.h>
#include "matx/matx_log.h"

matx_status_t matx_dense_z_i8_create(
	const matx_alloc_t* alloc,
	matx_dense_z_i8_t* out,
	matx_layout_t layout,
	matx_int64_t rows,
	matx_int64_t cols,
	matx_complex_d_t* data) {
	if (!out || !alloc || rows == 0 || cols == 0) {
		MATX_ERROR("%s: invalid argument", __func__);
		return MATX_ERR_INVALID_ARG;
	}
	if (layout != MATX_COL_MAJOR && layout != MATX_ROW_MAJOR) {
		MATX_ERROR("%s: invalid argument", __func__);
		return MATX_ERR_INVALID_ARG;
	}

	matx_dense_z_i8_opaque_t* out_value = matx_malloc(alloc, sizeof(matx_dense_z_i8_opaque_t));
	if (!out_value) {
		MATX_ERROR("%s: out of memory", __func__);
		return MATX_ERR_OUT_OF_MEMORY;
	}
	memset(out_value, 0, sizeof(*out_value));
	out_value->nrows = rows;
	out_value->ncols = cols;
	out_value->layout = layout;
	out_value->stride = (layout == MATX_COL_MAJOR) ? rows : cols;
	out_value->flags = 1u;

	const size_t n = rows * cols;
	out_value->data = (matx_complex_d_t*)matx_malloc(alloc, n * sizeof(matx_complex_d_t));
	if (!out_value->data) {
		matx_free(alloc, out_value);
		MATX_ERROR("%s: out of memory", __func__);
		return MATX_ERR_OUT_OF_MEMORY;
	}

	if (data != NULL)
	{
		memcpy(out_value->data, data, sizeof(matx_complex_d_t) * out_value->nrows * out_value->ncols);
	}
	if (*out != NULL)
	{
		matx_dense_z_i8_destroy(alloc, *out);
	}
	*out = out_value;
	return MATX_OK;
}

matx_status_t matx_dense_z_i8_dup(const matx_alloc_t* alloc, const matx_dense_z_i8_t in, matx_dense_z_i8_t* out)
{
	return matx_dense_z_i8_create(alloc, out, in->layout, in->nrows, in->ncols, in->data);
}

matx_status_t matx_dense_z_i8_wrap(
	const matx_alloc_t* alloc,
	matx_dense_z_i8_t* out,
	matx_int64_t rows,
	matx_int64_t cols,
	matx_int64_t stride,
	matx_layout_t layout,
	matx_complex_d_t* data) {
	if (!out || !data || rows == 0 || cols == 0) {
		MATX_ERROR("%s: invalid argument", __func__);
		return MATX_ERR_INVALID_ARG;
	}
	if (layout != MATX_COL_MAJOR && layout != MATX_ROW_MAJOR) {
		MATX_ERROR("%s: invalid argument", __func__);
		return MATX_ERR_INVALID_ARG;
	}
	if (layout == MATX_COL_MAJOR) {
		if (stride < rows) {
			MATX_ERROR("%s: invalid argument", __func__);
			return MATX_ERR_INVALID_ARG;
		}
	}
	else {
		if (stride < cols) {
			MATX_ERROR("%s: invalid argument", __func__);
			return MATX_ERR_INVALID_ARG;
		}
	}

	matx_dense_z_i8_opaque_t* out_value = matx_malloc(alloc, sizeof(matx_dense_z_i8_opaque_t));
	if (!out_value) {
		MATX_ERROR("%s: out of memory", __func__);
		return MATX_ERR_OUT_OF_MEMORY;
	}
	memset(out_value, 0, sizeof(*out_value));

	out_value->nrows = rows;
	out_value->ncols = cols;
	out_value->stride = stride;
	out_value->layout = layout;
	out_value->data = data;
	out_value->flags = 0u;
	if (*out != NULL)
	{
		matx_dense_z_i8_destroy(alloc, *out);
	}
	*out = out_value;
	return MATX_OK;
}

void matx_dense_z_i8_destroy(const matx_alloc_t* alloc, matx_dense_z_i8_t m) {
	if (!m) return;
	if ((m->flags & 1u) != 0u && m->data && alloc) {
		matx_free(alloc, m->data);
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

matx_status_t matx_dense_z_i8_fill(matx_dense_z_i8_t m, matx_complex_d_t val) {
    if (!m || !m->data) {
    	MATX_ERROR("%s: invalid argument", __func__);
    	return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t total = m->nrows * m->ncols;
    for (matx_int64_t i = 0; i < total; ++i)
        m->data[i] = val;
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_zeros(matx_dense_z_i8_t m) {
    if (!m || !m->data) {
    	MATX_ERROR("%s: invalid argument", __func__);
    	return MATX_ERR_INVALID_ARG;
    }
    memset(m->data, 0, (size_t)(m->nrows * m->ncols) * sizeof(matx_complex_d_t));
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_trace(const matx_dense_z_i8_t A, matx_complex_d_t* out) {
    if (!A || !out) {
    	MATX_ERROR("%s: invalid argument", __func__);
    	return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != A->ncols) {
    	MATX_ERROR("%s: invalid argument", __func__);
    	return MATX_ERR_INVALID_ARG;
    }
    matx_complex_d_t sum = {0.0, 0.0};
    for (matx_int64_t i = 0; i < A->nrows; ++i) {
        matx_int64_t idx = (A->layout == MATX_COL_MAJOR) ? i + i * A->stride : i * A->stride + i;
        sum.real += A->data[idx].real;
        sum.imag += A->data[idx].imag;
    }
    *out = sum;
    return MATX_OK;
}

matx_status_t matx_vec_d_i8_to_z_i8(const matx_alloc_t* alloc, const matx_vec_d_i8_t v, matx_vec_z_i8_t* out) {
    if (!alloc || !v || !out) {
    	MATX_ERROR("%s: invalid argument", __func__);
    	return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_vec_z_i8_create(alloc, out, NULL, v->n);
    if (st != MATX_OK) return st;
    for (matx_int64_t i = 0; i < v->n; ++i) {
        (*out)->data[i].real = v->data[i * v->stride];
        (*out)->data[i].imag = 0.0;
    }
    return MATX_OK;
}