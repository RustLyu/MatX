#ifndef MATX_READ_H
#define MATX_READ_H

#include "matx/matx_types.h"

#ifdef __cplusplus
extern "C" {
#endif
MATX_IO_API matx_status_t matx_read_dense_mtx_d_i8(const matx_alloc_t* alloc,
                                                matx_dense_d_i8_t* mtx,
                                                const char* file);
MATX_IO_API matx_status_t matx_read_dense_mtx_z_i8(const matx_alloc_t* alloc,
                                                matx_dense_z_i8_t* mtx,
                                                const char* file);

MATX_IO_API matx_status_t matx_read_sparse_mtx_d_i8(const matx_alloc_t* alloc,
                                                 matx_coo_d_i8_t* mtx,
                                                 const char* file);
MATX_IO_API matx_status_t matx_read_sparse_mtx_z_i8(const matx_alloc_t* alloc,
                                                 matx_coo_z_i8_t* mtx,
                                                 const char* file);

MATX_IO_API matx_status_t matx_read_vec_d_i8(const matx_alloc_t* alloc,
                                          matx_vec_d_i8_t* vec,
                                          const char* file);
MATX_IO_API matx_status_t matx_read_vec_z_i8(const matx_alloc_t* alloc,
                                          matx_vec_z_i8_t* vec,
                                          const char* file);

/* ---- CSV read ---- */

/**
 * @brief Read a real dense matrix from a CSV file (comma-separated values)
 * @param alloc Allocator
 * @param mtx Output real dense matrix (column-major)
 * @param file Input CSV file path
 * @return matx_status_t
 */
MATX_IO_API matx_status_t matx_read_csv_d_i8(const matx_alloc_t* alloc,
                                        matx_dense_d_i8_t* mtx,
                                        const char* file);

/**
 * @brief Read a complex dense matrix from a CSV file (two columns per element: real, imag)
 * @param alloc Allocator
 * @param mtx Output complex dense matrix (column-major)
 * @param file Input CSV file path
 * @return matx_status_t
 */
MATX_IO_API matx_status_t matx_read_csv_z_i8(const matx_alloc_t* alloc,
                                        matx_dense_z_i8_t* mtx,
                                        const char* file);

#ifdef __cplusplus
}
#endif

#endif