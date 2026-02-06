#include "matx/matx_compute.h"

#include <limits.h>
#include <stddef.h>

#if defined(MATX_HAVE_OPENBLAS) || defined(MATX_HAVE_BLIS)
#include <cblas.h>
#endif

matx_status_t matx_axpy_c64(const matx_blas_t* blas, 
                            matx_complex_f64 alpha,
                            const matx_vec_c64_t* x,
                            matx_vec_c64_t* y) {
  if (!x || !y || !x->data || !y->data)
    return MATX_ERR_INVALID_ARG;
  if (x->n != y->n)
    return MATX_ERR_INVALID_ARG;

  if (x->n > (size_t)INT_MAX || x->stride > (size_t)INT_MAX ||
      y->stride > (size_t)INT_MAX) {
    return MATX_ERR_NOT_SUPPORTED;
  }
  blas->vt.zaxpy((int)x->n, (const void*)&alpha, (const void*)x->data,
              (int)x->stride, (void*)y->data, (int)y->stride);
  return MATX_OK;
}
