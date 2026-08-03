#pragma once

#include "matx/matx_func.h"
#include "matx/matx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum matx_vec_backend_kind_t {
    MATX_VEC_BACKEND_REFERENCE = 0,
    MATX_VEC_BACKEND_OPENBLAS = 1,
    MATX_VEC_BACKEND_BLIS = 2,
} matx_vec_backend_kind_t;

typedef struct matx_vec_vtable_t
{
    // ---- Level 1: vector ops ----
    matx_status_t (*dscal)(matx_int64_t n, matx_double alpha, matx_double* x, matx_int64_t incx);
    matx_status_t (*zscal)(matx_int64_t n, const void* alpha, void* x, matx_int64_t incx);
    matx_status_t (*dcopy)(
        matx_int64_t n, const matx_double* x, matx_int64_t incx, matx_double* y, matx_int64_t incy);
    matx_status_t (*zcopy)(
        matx_int64_t n, const void* x, matx_int64_t incx, void* y, matx_int64_t incy);
    matx_status_t (*dswap)(
        matx_int64_t n, matx_double* x, matx_int64_t incx, matx_double* y, matx_int64_t incy);
    matx_status_t (*zswap)(matx_int64_t n, void* x, matx_int64_t incx, void* y, matx_int64_t incy);
    matx_status_t (*ddot)(matx_int64_t n,
                          const matx_double* x,
                          matx_int64_t incx,
                          const matx_double* y,
                          matx_int64_t incy,
                          matx_double* result);
    matx_status_t (*zdotu)(matx_int64_t n,
                           const void* x,
                           matx_int64_t incx,
                           const void* y,
                           matx_int64_t incy,
                           void* result);
    matx_status_t (*zdotc)(matx_int64_t n,
                           const void* x,
                           matx_int64_t incx,
                           const void* y,
                           matx_int64_t incy,
                           void* result);
    matx_status_t (*dnrm2)(matx_int64_t n,
                           const matx_double* x,
                           matx_int64_t incx,
                           matx_double* result);
    matx_status_t (*dznrm2)(matx_int64_t n, const void* x, matx_int64_t incx, matx_double* result);
    matx_status_t (*dasum)(matx_int64_t n,
                           const matx_double* x,
                           matx_int64_t incx,
                           matx_double* result);
    matx_status_t (*dzasum)(matx_int64_t n, const void* x, matx_int64_t incx, matx_double* result);
    matx_status_t (*idamax)(matx_int64_t n,
                            const matx_double* x,
                            matx_int64_t incx,
                            matx_int64_t* result);
    matx_status_t (*izamax)(matx_int64_t n, const void* x, matx_int64_t incx, matx_int64_t* result);
    matx_status_t (*daxpy)(matx_int64_t n,
                           matx_double alpha,
                           const matx_double* x,
                           matx_int64_t lda,
                           void* y,
                           matx_int64_t ldy);
    matx_status_t (*zaxpy)(matx_int64_t n,
                           const void* alpha,
                           const void* x,
                           matx_int64_t lda,
                           void* y,
                           matx_int64_t ldy);

    // ---- Vector norms ----
    matx_status_t (*norm1_d_i8)(matx_vec_d_i8_t A, matx_double* out);
    matx_status_t (*norm1_z_i8)(matx_vec_z_i8_t A, matx_double* out);
    matx_status_t (*norm2_d_i8)(matx_vec_d_i8_t A, matx_double* out);
    matx_status_t (*norm2_z_i8)(matx_vec_z_i8_t A, matx_double* out);
    matx_status_t (*norminf_d_i8)(matx_vec_d_i8_t A, matx_double* out);
    matx_status_t (*norminf_z_i8)(matx_vec_z_i8_t A, matx_double* out);

    // ---- Cross product ----
    matx_status_t (*cross_d_i8)(matx_vec_d_i8_t x, matx_vec_d_i8_t y, matx_vec_d_i8_t out);
    matx_status_t (*cross_z_i8)(matx_vec_z_i8_t x, matx_vec_z_i8_t y, matx_vec_z_i8_t out);
} matx_vec_vtable_t;

typedef struct matx_vec_backend_t
{
    matx_vec_backend_kind_t kind;
    matx_vec_vtable_t vt;
} matx_vec_backend_t;

MATX_API matx_vec_backend_t matx_vec_default(void);
MATX_API const char* matx_vec_backend_name(matx_vec_backend_kind_t k);

// ---- Level 1: scale / copy / swap / dot / nrm2 / asum / iamax / axpy ----

/**
	 * @brief Scale a real vector by a scalar (DSCAL)
	 * @formula x[i] := alpha * x[i],  i = 0, 1, ..., n-1
	 */
MATX_API matx_status_t matx_vec_scal_d_i8(const matx_vec_backend_t* blas,
                                          matx_double alpha,
                                          matx_vec_d_i8_t x);

/**
	 * @brief Scale a complex vector by a scalar (ZSCAL)
	 * @formula x[i] := alpha * x[i],  i = 0, 1, ..., n-1
	 */
MATX_API matx_status_t matx_vec_scal_z_i8(const matx_vec_backend_t* blas,
                                          matx_complex_d_t alpha,
                                          matx_vec_z_i8_t x);

/**
	 * @brief Copy a real vector (DCOPY)
	 * @formula y[i] := x[i],  i = 0, 1, ..., n-1
	 */
MATX_API matx_status_t matx_vec_copy_d_i8(const matx_vec_backend_t* blas,
                                          const matx_vec_d_i8_t x,
                                          matx_vec_d_i8_t y);

/**
	 * @brief Copy a complex vector (ZCOPY)
	 * @formula y[i] := x[i],  i = 0, 1, ..., n-1
	 */
MATX_API matx_status_t matx_vec_copy_z_i8(const matx_vec_backend_t* blas,
                                          const matx_vec_z_i8_t x,
                                          matx_vec_z_i8_t y);

/**
	 * @brief Swap two real vectors (DSWAP)
	 * @formula (x[i], y[i]) := (y[i], x[i]),  i = 0, 1, ..., n-1
	 */
MATX_API matx_status_t matx_vec_swap_d_i8(const matx_vec_backend_t* blas,
                                          matx_vec_d_i8_t x,
                                          matx_vec_d_i8_t y);

/**
	 * @brief Swap two complex vectors (ZSWAP)
	 * @formula (x[i], y[i]) := (y[i], x[i]),  i = 0, 1, ..., n-1
	 */
MATX_API matx_status_t matx_vec_swap_z_i8(const matx_vec_backend_t* blas,
                                          matx_vec_z_i8_t x,
                                          matx_vec_z_i8_t y);

/**
	 * @brief Dot product of two real vectors (DDOT)
	 * @formula result := x^T * y = sum_{i=0}^{n-1} x[i] * y[i]
	 */
MATX_API matx_status_t matx_vec_dot_d_i8(const matx_vec_backend_t* blas,
                                         const matx_vec_d_i8_t x,
                                         const matx_vec_d_i8_t y,
                                         matx_double* result);

/**
	 * @brief Unconjugated dot product of two complex vectors (ZDOTU)
	 * @formula result := x^T * y = sum_{i=0}^{n-1} x[i] * y[i]  (no conjugation)
	 */
MATX_API matx_status_t matx_vec_dotu_z_i8(const matx_vec_backend_t* blas,
                                          const matx_vec_z_i8_t x,
                                          const matx_vec_z_i8_t y,
                                          matx_complex_d_t* result);

/**
	 * @brief Conjugated dot product of two complex vectors (ZDOTC)
	 * @formula result := x^H * y = sum_{i=0}^{n-1} conj(x[i]) * y[i]
	 */
MATX_API matx_status_t matx_vec_dotc_z_i8(const matx_vec_backend_t* blas,
                                          const matx_vec_z_i8_t x,
                                          const matx_vec_z_i8_t y,
                                          matx_complex_d_t* result);

/**
	 * @brief Euclidean norm of a real vector (DNRM2)
	 * @formula result := ||x||_2 = sqrt( sum_{i=0}^{n-1} x[i]^2 )
	 */
MATX_API matx_status_t matx_vec_nrm2_d_i8(const matx_vec_backend_t* blas,
                                          const matx_vec_d_i8_t x,
                                          matx_double* result);

/**
	 * @brief Euclidean norm of a complex vector (DZNRM2)
	 * @formula result := ||x||_2 = sqrt( sum_{i=0}^{n-1} |x[i]|^2 )
	 */
MATX_API matx_status_t matx_vec_nrm2_z_i8(const matx_vec_backend_t* blas,
                                          const matx_vec_z_i8_t x,
                                          matx_double* result);

/**
	 * @brief Sum of absolute values of a real vector (DASUM)
	 * @formula result := sum_{i=0}^{n-1} |x[i]|
	 */
MATX_API matx_status_t matx_vec_asum_d_i8(const matx_vec_backend_t* blas,
                                          const matx_vec_d_i8_t x,
                                          matx_double* result);

/**
	 * @brief Sum of absolute values of real and imaginary parts of a complex vector (DZASUM)
	 * @formula result := sum_{i=0}^{n-1} ( |Re(x[i])| + |Im(x[i])| )
	 */
MATX_API matx_status_t matx_vec_asum_z_i8(const matx_vec_backend_t* blas,
                                          const matx_vec_z_i8_t x,
                                          matx_double* result);

/**
	 * @brief Index of element with max absolute value in a real vector (IDAMAX)
	 * @formula result := argmax_i |x[i]|,  i = 0, 1, ..., n-1
	 */
MATX_API matx_status_t matx_vec_iamax_d_i8(const matx_vec_backend_t* blas,
                                           const matx_vec_d_i8_t x,
                                           matx_int64_t* result);

/**
	 * @brief Index of element with max absolute value in a complex vector (IZAMAX)
	 * @formula result := argmax_i |x[i]|,  i = 0, 1, ..., n-1
	 */
MATX_API matx_status_t matx_vec_iamax_z_i8(const matx_vec_backend_t* blas,
                                           const matx_vec_z_i8_t x,
                                           matx_int64_t* result);

/**
	 * @brief Real vector scaled accumulation (DAXPY)
	 * @formula y[i] := alpha * x[i] + y[i],  i = 0, 1, ..., n-1
	 */
MATX_API matx_status_t matx_vec_axpy_d_i8(const matx_vec_backend_t* blas,
                                          matx_double alpha,
                                          const matx_vec_d_i8_t x,
                                          matx_vec_d_i8_t y);

/**
	 * @brief Complex vector scaled accumulation (ZAXPY)
	 * @formula y[i] := alpha * x[i] + y[i],  i = 0, 1, ..., n-1
	 */
MATX_API matx_status_t matx_vec_axpy_z_i8(const matx_vec_backend_t* blas,
                                          matx_complex_d_t alpha,
                                          const matx_vec_z_i8_t x,
                                          matx_vec_z_i8_t y);

// ---- Vector norms ----

/**
	 * @brief 1-norm of a real vector
	 * @formula ||v||_1 = sum_{i=0}^{n-1} |v[i]|
	 */
MATX_API matx_status_t matx_vec_norm1_d_i8(const matx_vec_backend_t* backend,
                                           matx_vec_d_i8_t A,
                                           matx_double* out);

/**
	 * @brief 1-norm of a complex vector
	 * @formula ||v||_1 = sum_{i=0}^{n-1} |v[i]|
	 */
MATX_API matx_status_t matx_vec_norm1_z_i8(const matx_vec_backend_t* backend,
                                           matx_vec_z_i8_t A,
                                           matx_double* out);

/**
	 * @brief 2-norm (Euclidean norm) of a real vector
	 * @formula ||v||_2 = sqrt( sum_{i=0}^{n-1} v[i]^2 )
	 */
MATX_API matx_status_t matx_vec_norm2_d_i8(const matx_vec_backend_t* backend,
                                           matx_vec_d_i8_t A,
                                           matx_double* out);

/**
	 * @brief 2-norm (Euclidean norm) of a complex vector
	 * @formula ||v||_2 = sqrt( sum_{i=0}^{n-1} |v[i]|^2 )
	 */
MATX_API matx_status_t matx_vec_norm2_z_i8(const matx_vec_backend_t* backend,
                                           matx_vec_z_i8_t A,
                                           matx_double* out);

/**
	 * @brief Infinity-norm of a real vector
	 * @formula ||v||_inf = max_{i=0,...,n-1} |v[i]|
	 */
MATX_API matx_status_t matx_vec_norminf_d_i8(const matx_vec_backend_t* backend,
                                             matx_vec_d_i8_t A,
                                             matx_double* out);

/**
	 * @brief Infinity-norm of a complex vector
	 * @formula ||v||_inf = max_{i=0,...,n-1} |v[i]|
	 */
MATX_API matx_status_t matx_vec_norminf_z_i8(const matx_vec_backend_t* backend,
                                             matx_vec_z_i8_t A,
                                             matx_double* out);

// ---- Cross product ----

/**
	 * @brief Cross product of two real 3D vectors
	 * @formula out = x × y
	 *          out[0] = x[1]*y[2] - x[2]*y[1]
	 *          out[1] = x[2]*y[0] - x[0]*y[2]
	 *          out[2] = x[0]*y[1] - x[1]*y[0]
	 *          All vectors must have length 3.
	 */
MATX_API matx_status_t matx_vec_cross_d_i8(const matx_vec_backend_t* blas,
                                           matx_vec_d_i8_t x,
                                           matx_vec_d_i8_t y,
                                           matx_vec_d_i8_t out);

/**
	 * @brief Cross product of two complex 3D vectors
	 * @formula out = x × y
	 *          All vectors must have length 3.
	 */
MATX_API matx_status_t matx_vec_cross_z_i8(const matx_vec_backend_t* blas,
                                           matx_vec_z_i8_t x,
                                           matx_vec_z_i8_t y,
                                           matx_vec_z_i8_t out);

#ifdef __cplusplus
}
#endif