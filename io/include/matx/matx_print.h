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

/* ---- Matrix info / summary ---- */

/**
 * @brief Format a human-readable summary of a real dense matrix into a buffer
 * @param mtx Dense real matrix
 * @param buf Output buffer (caller-allocated)
 * @param bufsize Size of output buffer
 * @return Number of characters written (excluding null terminator), or negative on error
 */
MATX_IO_API int matx_dense_d_i8_info(const matx_dense_d_i8_t mtx, char* buf, size_t bufsize);

/**
 * @brief Format a human-readable summary of a complex dense matrix into a buffer
 * @param mtx Dense complex matrix
 * @param buf Output buffer (caller-allocated)
 * @param bufsize Size of output buffer
 * @return Number of characters written (excluding null terminator), or negative on error
 */
MATX_IO_API int matx_dense_z_i8_info(const matx_dense_z_i8_t mtx, char* buf, size_t bufsize);

/**
 * @brief Format a human-readable summary of a real sparse (COO) matrix into a buffer
 * @param mtx COO sparse real matrix
 * @param buf Output buffer (caller-allocated)
 * @param bufsize Size of output buffer
 * @return Number of characters written (excluding null terminator), or negative on error
 */
MATX_IO_API int matx_coo_d_i8_info(const matx_coo_d_i8_t mtx, char* buf, size_t bufsize);

/**
 * @brief Format a human-readable summary of a complex sparse (COO) matrix into a buffer
 * @param mtx COO sparse complex matrix
 * @param buf Output buffer (caller-allocated)
 * @param bufsize Size of output buffer
 * @return Number of characters written (excluding null terminator), or negative on error
 */
MATX_IO_API int matx_coo_z_i8_info(const matx_coo_z_i8_t mtx, char* buf, size_t bufsize);

#ifdef __cplusplus
}
#endif

#endif