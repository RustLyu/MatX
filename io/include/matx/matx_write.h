#ifndef MATX_WRITE_H
#define MATX_WRITE_H

#include "matx/matx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Write a real dense matrix to a Matrix Market (.mtx) array file
 * @param mtx Dense real matrix
 * @param file Output file path
 */
MATX_IO_API void matx_write_dense_mtx_d_i8(const matx_dense_d_i8_t mtx, const char* file);

/**
 * @brief Write a complex dense matrix to a Matrix Market (.mtx) array file
 * @param mtx Dense complex matrix
 * @param file Output file path
 */
MATX_IO_API void matx_write_dense_mtx_z_i8(const matx_dense_z_i8_t mtx, const char* file);

/**
 * @brief Write a real sparse (COO) matrix to a Matrix Market (.mtx) coordinate file
 * @param mtx COO sparse real matrix
 * @param file Output file path
 */
MATX_IO_API void matx_write_sparse_mtx_d_i8(const matx_coo_d_i8_t mtx, const char* file);

/**
 * @brief Write a complex sparse (COO) matrix to a Matrix Market (.mtx) coordinate file
 * @param mtx COO sparse complex matrix
 * @param file Output file path
 */
MATX_IO_API void matx_write_sparse_mtx_z_i8(const matx_coo_z_i8_t mtx, const char* file);

/**
 * @brief Write a real vector to a Matrix Market (.mtx) array file (n x 1 column vector)
 * @param vec Real vector
 * @param file Output file path
 */
MATX_IO_API void matx_write_vec_d_i8(const matx_vec_d_i8_t vec, const char* file);

/**
 * @brief Write a complex vector to a Matrix Market (.mtx) array file (n x 1 column vector)
 * @param vec Complex vector
 * @param file Output file path
 */
MATX_IO_API void matx_write_vec_z_i8(const matx_vec_z_i8_t vec, const char* file);

/* ---- CSV write ---- */

/**
 * @brief Write a real dense matrix to a CSV file
 * @param mtx Dense real matrix
 * @param file Output file path
 */
MATX_IO_API void matx_write_csv_d_i8(const matx_dense_d_i8_t mtx, const char* file);

/**
 * @brief Write a complex dense matrix to a CSV file (two columns per element: real, imag)
 * @param mtx Dense complex matrix
 * @param file Output file path
 */
MATX_IO_API void matx_write_csv_z_i8(const matx_dense_z_i8_t mtx, const char* file);

#ifdef __cplusplus
}
#endif

#endif // MATX_WRITE_H