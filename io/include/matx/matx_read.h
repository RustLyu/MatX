#ifndef MATX_READ_H
#define MATX_READ_H

#include "matx/matx_types.h"

#ifdef __cplusplus
extern "C" {
#endif
    MATX_API matx_status_t matx_read_dense_mtx_f64(matx_dense_f64_t* mtx, const char* file, const matx_alloc_t* alloc);
    MATX_API matx_status_t matx_read_dense_mtx_c64(matx_dense_c64_t* mtx, const char* file, const matx_alloc_t* alloc);

    MATX_API matx_status_t matx_read_sparse_mtx_f64(matx_coo_f64_t* mtx, const char* file, const matx_alloc_t* alloc);
    MATX_API matx_status_t matx_read_sparse_mtx_c64(matx_coo_c64_t* mtx, const char* file, const matx_alloc_t* alloc);

    MATX_API matx_status_t matx_read_vec_f64(matx_vec_f64_t* vec, const char* file, const matx_alloc_t* alloc);
    MATX_API matx_status_t matx_read_vec_c64(matx_vec_c64_t* vec, const char* file, const matx_alloc_t* alloc);

#ifdef __cplusplus
}
#endif

#endif