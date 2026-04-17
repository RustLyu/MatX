#include "matx/matx_types.h"
#include "matx/matx_func.h"
#include <string.h>

matx_status_t matx_vec_f64_create(
    const matx_alloc_t* alloc,
    matx_vec_f64_t* out,
    matx_double* data,
    matx_int64_t n) {
  if (!out || !alloc || n == 0) return MATX_ERR_INVALID_ARG;
  memset(out, 0, sizeof(*out));
  out->n = n;
  out->stride = 1;
  out->flags = 1u; // owns data
  out->data = (matx_double*)matx_malloc(alloc, n * sizeof(matx_double));
  if (!out->data) {
    memset(out, 0, sizeof(*out));
    return MATX_ERR_OUT_OF_MEMORY;
  }
  if (data != NULL)
  {
      memcpy(out->data, data, sizeof(matx_double) * out->n);
  }
  return MATX_OK;
}

matx_status_t matx_vec_f64_wrap(matx_vec_f64_t* out,
    matx_int64_t n,
    matx_int64_t stride,
    matx_double* data) {
  if (!out || !data || n == 0 || stride == 0) return MATX_ERR_INVALID_ARG;
  out->n = n;
  out->stride = stride;
  out->data = data;
  out->flags = 0u;
  return MATX_OK;
}

void matx_vec_f64_destroy(const matx_alloc_t* alloc, matx_vec_f64_t* v) {
  if (!v) return;
  if ((v->flags & 1u) != 0u && v->data && alloc) {
    matx_free(alloc, v->data);
  }
  memset(v, 0, sizeof(*v));
}

matx_status_t matx_vec_c64_create(const matx_alloc_t* alloc, 
    matx_vec_c64_t* out,
    matx_complex_f64* data,
    matx_int64_t n) {
  if (!out || !alloc || n == 0) return MATX_ERR_INVALID_ARG;
  memset(out, 0, sizeof(*out));
  out->n = n;
  out->stride = 1;
  out->flags = 1u;
  out->data = (matx_complex_f64*)matx_malloc(alloc, n * sizeof(matx_complex_f64));
  if (!out->data) {
    memset(out, 0, sizeof(*out));
    return MATX_ERR_OUT_OF_MEMORY;
  }

  if (data != NULL)
  {
      memcpy(out->data, data, sizeof(matx_complex_f64) * out->n);
  }

  return MATX_OK;
}

matx_status_t matx_vec_c64_wrap(matx_vec_c64_t* out,
    matx_int64_t n,
    matx_int64_t stride,
                                matx_complex_f64* data) {
  if (!out || !data || n == 0 || stride == 0) return MATX_ERR_INVALID_ARG;
  out->n = n;
  out->stride = stride;
  out->data = data;
  out->flags = 0u;
  return MATX_OK;
}

void matx_vec_c64_destroy(const matx_alloc_t* alloc, matx_vec_c64_t* v) {
  if (!v) return;
  if ((v->flags & 1u) != 0u && v->data && alloc) {
    matx_free(alloc, v->data);
  }

  if (v->handle_grb.valid > 0)
  {
      if (v->handle_grb.custom_free_func && v->handle_grb.impl)
      {
          v->handle_grb.custom_free_func(v->handle_grb.impl);
      }
      v->handle_grb.impl = NULL;
      v->handle_grb.valid = -1;
  }

  memset(v, 0, sizeof(*v));
}

