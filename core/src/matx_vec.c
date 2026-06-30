#include "matx/matx_types.h"
#include "matx/matx_func.h"
#include "matx/matx_types_internal.h"

#include <string.h>

matx_status_t matx_vec_d_i8_create(
    const matx_alloc_t* alloc,
    matx_vec_d_i8_t* out,
    matx_double* data,
    matx_int64_t n) {
  if (!out || !alloc || n == 0) return MATX_ERR_INVALID_ARG;

  matx_vec_d_i8_opaque_t* out_value = matx_malloc(alloc, sizeof(matx_vec_d_i8_opaque_t));
  memset(out_value, 0, sizeof(matx_vec_d_i8_opaque_t));
  out_value->n = n;
  out_value->stride = 1;
  out_value->flags = 1u; // owns data
  out_value->data = (matx_double*)matx_malloc(alloc, n * sizeof(matx_double));

  if (!out_value->data) {
      matx_free(alloc, out_value);
    return MATX_ERR_OUT_OF_MEMORY;
  }
  if (data != NULL)
  {
      memcpy(out_value->data, data, sizeof(matx_double) * out_value->n);
  }
  if (*out != NULL)
  {
      matx_vec_d_i8_destroy(alloc, *out);
  }
  *out = out_value;
  return MATX_OK;
}

matx_status_t matx_vec_d_i8_dup(const matx_alloc_t* alloc, const matx_vec_d_i8_t in, matx_vec_d_i8_t* out)
{
    return matx_vec_d_i8_create(alloc, out, in->data, in->n);
}

matx_status_t matx_vec_d_i8_wrap(const matx_alloc_t* alloc, matx_vec_d_i8_t* out,
    matx_int64_t n,
    matx_int64_t stride,
    matx_double* data) {
  if (!out || !data || n == 0 || stride == 0)
return MATX_ERR_INVALID_ARG;

  matx_vec_d_i8_opaque_t* out_value = matx_malloc(alloc, sizeof(matx_vec_d_i8_opaque_t));
  memset(out_value, 0, sizeof(matx_vec_d_i8_opaque_t));

  out_value->n = n;
  out_value->stride = stride;
  out_value->data = data;
  out_value->flags = 0u;
  if (*out != NULL)
  {
      matx_vec_d_i8_destroy(alloc, *out);
  }
  *out = out_value;
  return MATX_OK;
}

void matx_vec_d_i8_destroy(const matx_alloc_t* alloc, matx_vec_d_i8_t v) {
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
  matx_free(alloc, v);
}

matx_status_t matx_vec_z_i8_create(const matx_alloc_t* alloc, 
    matx_vec_z_i8_t* out,
    matx_complex_d_i8_t* data,
    matx_int64_t n) {
  if (!out || !alloc || n == 0) return MATX_ERR_INVALID_ARG;
  matx_vec_z_i8_opaque_t* out_value = matx_malloc(alloc, sizeof(matx_vec_z_i8_opaque_t));
  memset(out_value, 0, sizeof(matx_vec_z_i8_opaque_t));

  out_value->n = n;
  out_value->stride = 1;
  out_value->flags = 1u;
  out_value->data = (matx_complex_d_i8_t*)matx_malloc(alloc, n * sizeof(matx_complex_d_i8_t));
  if (!out_value->data) {
    matx_free(alloc, out_value);
    return MATX_ERR_OUT_OF_MEMORY;
  }

  if (data != NULL)
  {
      memcpy(out_value->data, data, sizeof(matx_complex_d_i8_t) * out_value->n);
  }
  if (*out != NULL)
  {
      matx_vec_z_i8_destroy(alloc, *out);
  }
  *out = out_value;
  return MATX_OK;
}

matx_status_t matx_vec_z_i8_dup(const matx_alloc_t* alloc, matx_vec_z_i8_t in, matx_vec_z_i8_t* out)
{
    return matx_vec_z_i8_create(alloc, out, in->data, in->n);
}

matx_status_t matx_vec_z_i8_wrap(const matx_alloc_t* alloc, matx_vec_z_i8_t* out,
    matx_int64_t n,
    matx_int64_t stride,
                                matx_complex_d_i8_t* data) {
  if (!out || !data || n == 0 || stride == 0) return MATX_ERR_INVALID_ARG;

  matx_vec_z_i8_opaque_t* out_value = matx_malloc(alloc, sizeof(matx_vec_z_i8_opaque_t));
  memset(out_value, 0, sizeof(matx_vec_z_i8_opaque_t));

  out_value->n = n;
  out_value->stride = stride;
  out_value->data = data;
  out_value->flags = 0u;
  if (*out != NULL)
  {
      matx_vec_z_i8_destroy(alloc, *out);
  }
  *out = out_value;
  return MATX_OK;
}

void matx_vec_z_i8_destroy(const matx_alloc_t* alloc, matx_vec_z_i8_t v) {
  if (!v) return;
  if ((v->flags & 1u) != 0u && v->data && alloc) {
    matx_free(alloc, v->data);
v->data = NULL;
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
  matx_free(alloc, v);
  //memset(v, 0, sizeof(*v));
}

matx_status_t matx_vec_d_i8_fill(matx_vec_d_i8_t v, matx_double val) {
    if (!v || !v->data) return MATX_ERR_INVALID_ARG;
    for (matx_int64_t i = 0; i < v->n; ++i)
        v->data[i * v->stride] = val;
    return MATX_OK;
}

matx_status_t matx_vec_z_i8_fill(matx_vec_z_i8_t v, matx_complex_d_i8_t val) {
    if (!v || !v->data) return MATX_ERR_INVALID_ARG;
    for (matx_int64_t i = 0; i < v->n; ++i)
        v->data[i * v->stride] = val;
    return MATX_OK;
}

matx_status_t matx_vec_d_i8_zeros(matx_vec_d_i8_t v) {
    if (!v || !v->data) return MATX_ERR_INVALID_ARG;
    if (v->stride == 1)
        memset(v->data, 0, v->n * sizeof(matx_double));
    else
        for (matx_int64_t i = 0; i < v->n; ++i)
            v->data[i * v->stride] = 0.0;
    return MATX_OK;
}

matx_status_t matx_vec_z_i8_zeros(matx_vec_z_i8_t v) {
    if (!v || !v->data) return MATX_ERR_INVALID_ARG;
    if (v->stride == 1)
        memset(v->data, 0, v->n * sizeof(matx_complex_d_i8_t));
    else {
        matx_complex_d_i8_t zero = {0.0, 0.0};
        for (matx_int64_t i = 0; i < v->n; ++i)
            v->data[i * v->stride] = zero;
    }
    return MATX_OK;
}

matx_status_t matx_vec_d_i8_ones(matx_vec_d_i8_t v) {
    return matx_vec_d_i8_fill(v, 1.0);
}

