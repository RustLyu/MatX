#include "matx/matx_compute.h"
#include "matx/matx_c64_utils.h"

#include <limits.h>
#include <stddef.h>

#if defined(MATX_HAVE_OPENBLAS) || defined(MATX_HAVE_BLIS)
#include <cblas.h>
#endif

matx_status_t matx_geadd_c64(matx_complex_f64 alpha,
                             const matx_dense_c64_t* A,
                             matx_complex_f64 beta,
                             matx_dense_c64_t* B) {
  if (!A || !B || !A->data || !B->data)
    return MATX_ERR_INVALID_ARG;
  if (A->rows != B->rows || A->cols != B->cols)
    return MATX_ERR_INVALID_ARG;
  if (A->layout != B->layout)
    return MATX_ERR_INVALID_ARG;

  const size_t m = A->rows;
  const size_t n = A->cols;

  if (A->layout == MATX_COL_MAJOR) {
    for (size_t j = 0; j < n; ++j) {
      for (size_t i = 0; i < m; ++i) {
        const size_t ia = i + j * A->stride;
        const size_t ib = i + j * B->stride;
        B->data[ib] =
            matx_c64_add(matx_c64_mul(alpha, A->data[ia]),
                         matx_c64_mul(beta, B->data[ib]));
      }
    }
  } else {
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
        const size_t ia = j + i * A->stride;
        const size_t ib = j + i * B->stride;
        B->data[ib] =
            matx_c64_add(matx_c64_mul(alpha, A->data[ia]),
                         matx_c64_mul(beta, B->data[ib]));
      }
    }
  }
  return MATX_OK;
}
