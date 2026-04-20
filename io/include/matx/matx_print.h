#ifndef MATX_PRINT_H
#define MATX_PRINT_H

#include "matx/matx_types.h"

#ifdef __cplusplus
extern "C" {
#endif
    MATX_API void matx_print_dense_mtx_f64(const matx_dense_f64_t mtx, const char* file);
    MATX_API void matx_print_dense_mtx_c64(const matx_dense_c64_t mtx, const char* file);

    MATX_API void matx_print_sparse_mtx_f64(const matx_coo_f64_t mtx, const char* file);
    MATX_API void matx_print_sparse_mtx_c64(const matx_coo_c64_t mtx, const char* file);

    MATX_API void matx_print_vec_f64(const matx_vec_f64_t vec, const char* file);
    MATX_API void matx_print_vec_c64(const matx_vec_c64_t vec, const char* file);

#ifdef __cplusplus
}
#endif

#endif