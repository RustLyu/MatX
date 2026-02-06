#include "matx/matx.h"

#include <string.h>

matx_status_t matx_vec_f64_create(matx_vec_f64_t* out,
                                  size_t n,
                                  const matx_alloc_t* alloc) {
  if (!out || !alloc || n == 0) return MATX_ERR_INVALID_ARG;
  memset(out, 0, sizeof(*out));
  out->n = n;
  out->stride = 1;
  out->flags = 1u; // owns data
  out->data = (double*)matx_malloc(alloc, n * sizeof(double));
  if (!out->data) {
    memset(out, 0, sizeof(*out));
    return MATX_ERR_OUT_OF_MEMORY;
  }
  return MATX_OK;
}

matx_status_t matx_vec_f64_wrap(matx_vec_f64_t* out,
                                size_t n,
                                size_t stride,
                                double* data) {
  if (!out || !data || n == 0 || stride == 0) return MATX_ERR_INVALID_ARG;
  out->n = n;
  out->stride = stride;
  out->data = data;
  out->flags = 0u;
  return MATX_OK;
}

void matx_vec_f64_destroy(matx_vec_f64_t* v,
                          const matx_alloc_t* alloc) {
  if (!v) return;
  if ((v->flags & 1u) != 0u && v->data && alloc) {
    matx_free(alloc, v->data);
  }
  memset(v, 0, sizeof(*v));
}

matx_status_t matx_vec_c64_create(matx_vec_c64_t* out,
                                  size_t n,
                                  const matx_alloc_t* alloc) {
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
  return MATX_OK;
}

matx_status_t matx_vec_c64_wrap(matx_vec_c64_t* out,
                                size_t n,
                                size_t stride,
                                matx_complex_f64* data) {
  if (!out || !data || n == 0 || stride == 0) return MATX_ERR_INVALID_ARG;
  out->n = n;
  out->stride = stride;
  out->data = data;
  out->flags = 0u;
  return MATX_OK;
}

void matx_vec_c64_destroy(matx_vec_c64_t* v,
                          const matx_alloc_t* alloc) {
  if (!v) return;
  if ((v->flags & 1u) != 0u && v->data && alloc) {
    matx_free(alloc, v->data);
  }
  memset(v, 0, sizeof(*v));
}

