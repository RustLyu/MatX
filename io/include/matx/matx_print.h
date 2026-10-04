#ifndef MATX_PRINT_H
#define MATX_PRINT_H

#include "matx/matx_types.h"

#ifdef __cplusplus
extern "C" {
#endif
MATX_IO_API void matx_print_dense_mtx_d_i8(const matx_dense_d_i8_t& mtx, const char* file);
MATX_IO_API void matx_print_dense_mtx_z_i8(const matx_dense_z_i8_t& mtx, const char* file);

MATX_IO_API void matx_print_sparse_mtx_d_i8(const matx_coo_d_i8_t& mtx, const char* file);
MATX_IO_API void matx_print_sparse_mtx_z_i8(const matx_coo_z_i8_t& mtx, const char* file);

MATX_IO_API void matx_print_vec_d_i8(const matx_vec_d_i8_t& vec, const char* file);
MATX_IO_API void matx_print_vec_z_i8(const matx_vec_z_i8_t& vec, const char* file);

#ifdef __cplusplus
}
#endif

#endif