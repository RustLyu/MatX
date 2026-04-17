#include "matx/matx_types.h"
#include "matx/matx_func.h"
#include <string.h>

matx_status_t matx_dense_c64_create(
    const matx_alloc_t* alloc,
    matx_dense_c64_t* out,
    matx_layout_t layout,
    matx_int64_t rows,
    matx_int64_t cols,
    matx_complex_f64* data) {
  if (!out || !alloc || rows == 0 || cols == 0) return MATX_ERR_INVALID_ARG;
  if (layout != MATX_COL_MAJOR && layout != MATX_ROW_MAJOR) return MATX_ERR_INVALID_ARG;

  memset(out, 0, sizeof(*out));
  out->nrows = rows;
  out->ncols = cols;
  out->layout = layout;
  out->stride = (layout == MATX_COL_MAJOR) ? rows : cols;
  out->flags = 1u;

  const size_t n = rows * cols;
  out->data = (matx_complex_f64*)matx_malloc(alloc, n * sizeof(matx_complex_f64));
  if (!out->data) {
    memset(out, 0, sizeof(*out));
    return MATX_ERR_OUT_OF_MEMORY;
  }

  if (data != NULL)
  {
      memcpy(out->data, data, sizeof(matx_complex_f64) * out->nrows * out->ncols);
  }

  return MATX_OK;
}

matx_status_t matx_dense_c64_dup(const matx_alloc_t* alloc, const matx_dense_c64_t* const in, matx_dense_c64_t* out)
{
    return matx_dense_c64_create(alloc, out, in->layout, in->nrows, in->ncols, in->data);
}

matx_status_t matx_dense_c64_wrap(matx_dense_c64_t* out,
                                  matx_int64_t rows,
    matx_int64_t cols,
    matx_int64_t stride,
                                  matx_layout_t layout,
                                  matx_complex_f64* data) {
  if (!out || !data || rows == 0 || cols == 0) return MATX_ERR_INVALID_ARG;
  if (layout != MATX_COL_MAJOR && layout != MATX_ROW_MAJOR) return MATX_ERR_INVALID_ARG;
  if (layout == MATX_COL_MAJOR) {
    if (stride < rows) return MATX_ERR_INVALID_ARG;
  } else {
    if (stride < cols) return MATX_ERR_INVALID_ARG;
  }

  out->nrows = rows;
  out->ncols = cols;
  out->stride = stride;
  out->layout = layout;
  out->data = data;
  out->flags = 0u;
  return MATX_OK;
}

void matx_dense_c64_destroy(const matx_alloc_t* alloc, matx_dense_c64_t* m) {
  if (!m) return;
  if ((m->flags & 1u) != 0u && m->data && alloc) {
    matx_free(alloc, m->data);
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

