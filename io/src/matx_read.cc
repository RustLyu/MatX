
#include "matx/matx_read.h"
#include "matx/matx_func.h"
#include "matx/matx_log.h"

#include <fstream>
#include <iostream>
#include <iomanip>

matx_status_t matx_read_dense_mtx_f64(const matx_alloc_t* alloc, matx_dense_f64_t* mtx, const char* file)
{
    matx_status_t ret = MATX_ERR_INTERNAL;
    std::ifstream is(file);
    if (!is.is_open())
    {
        MATX_ERROR("open file error. path:%f", file);
        return ret;
    }

    matx_int64_t m = -1;
    matx_int64_t n = -1;
    matx_int64_t layout = -1;
    is >> m >> n >> layout;

    matx_dense_f64_create(alloc, mtx, (matx_layout_t)layout, m, n, NULL);
    mtx->stride = (mtx->layout == MATX_ROW_MAJOR) ? mtx->ncols : mtx->nrows;
    for (matx_int64_t i = 0; i < mtx->nrows; ++i) {
        for (matx_int64_t j = 0; j < mtx->ncols; ++j) {
            matx_int64_t idx = (mtx->layout == MATX_ROW_MAJOR)
                ? i * mtx->stride + j
                : j * mtx->stride + i;
            is >> mtx->data[idx];
        }
    }

    is.close();
    ret = MATX_OK;
    return ret;
}

matx_status_t matx_read_dense_mtx_c64(const matx_alloc_t* alloc, matx_dense_c64_t* mtx, const char* file)
{
    matx_status_t ret = MATX_ERR_INTERNAL;
    std::ifstream is(file);
    if (!is.is_open())
    {
        MATX_ERROR("open file error. path:%f", file);
        return ret;
    }

    matx_int64_t m = -1;
    matx_int64_t n = -1;
    matx_int64_t layout = -1;
    is >> m >> n >> layout;

    matx_dense_c64_create(alloc, mtx, (matx_layout_t)layout, m, n, NULL);
    mtx->stride = (mtx->layout == MATX_ROW_MAJOR) ? mtx->ncols : mtx->nrows;
    for (matx_int64_t i = 0; i < mtx->nrows; ++i) {
        for (matx_int64_t j = 0; j < mtx->ncols; ++j) {
            matx_int64_t idx = (mtx->layout == MATX_ROW_MAJOR)
                ? i * mtx->stride + j
                : j * mtx->stride + i;
            is >> mtx->data[idx].real >> mtx->data[idx].imag;
        }
    }

    is.close();
    ret = MATX_OK;
    return ret;
}

matx_status_t matx_read_sparse_mtx_f64(const matx_alloc_t* alloc, matx_coo_f64_t* mtx, const char* file)
{
    matx_status_t ret = MATX_ERR_INTERNAL;
    std::ifstream is(file);
    if (!is.is_open())
    {
        MATX_ERROR("open file error. path:%f", file);
        return ret;
    }

    matx_int64_t m = -1;
    matx_int64_t n = -1;
    matx_int64_t nnz = -1;
    is >> m >> n >> nnz;

    matx_coo_sparse_f64_create(alloc, mtx, m, n, nnz, NULL, NULL, NULL);
    for (matx_int64_t i = 0; i < nnz; ++i) 
    {
        is >> mtx->rows[i] >> mtx->columns[i] >> mtx->values[i];
    }

    is.close();

    ret = MATX_OK;
    return ret;
}

matx_status_t matx_read_sparse_mtx_c64(const matx_alloc_t* alloc, matx_coo_c64_t* mtx, const char* file)
{
    matx_status_t ret = MATX_ERR_INTERNAL;
    std::ifstream is(file);
    if (!is.is_open())
    {
        MATX_ERROR("open file error. path:%f", file);
        return ret;
    }

    matx_int64_t m = -1;
    matx_int64_t n = -1;
    matx_int64_t nnz = -1;
    is >> m >> n >> nnz;

    matx_coo_sparse_c64_create(alloc, mtx, m, n, nnz, NULL, NULL, NULL);
    for (matx_int64_t i = 0; i < nnz; ++i)
    {
        is >> mtx->rows[i] >> mtx->columns[i] >> mtx->values[i].real >> mtx->values[i].imag;
    }

    is.close();

    ret = MATX_OK;
    return ret;
}

matx_status_t matx_read_vec_f64(const matx_alloc_t* alloc, matx_vec_f64_t* vec, const char* file)
{
    matx_status_t ret = MATX_ERR_INTERNAL;
    std::ifstream is(file);
    if (!is.is_open())
    {
        MATX_ERROR("open file error. path:%f", file);
        return ret;
    }

    matx_int64_t n = -1;
    is >> n;

    matx_vec_f64_create(alloc, vec, NULL, n);
    for (matx_int64_t i = 0; i < n; ++i)
    {
        is >> vec->data[i];
    }

    is.close();

    ret = MATX_OK;
    return ret;
}

matx_status_t matx_read_vec_c64(const matx_alloc_t* alloc, matx_vec_c64_t* vec, const char* file)
{
    matx_status_t ret = MATX_ERR_INTERNAL;
    std::ifstream is(file);
    if (!is.is_open())
    {
        MATX_ERROR("open file error. path:%f", file);
        return ret;
    }

    matx_int64_t n = -1;
    is >> n;

    matx_vec_c64_create(alloc, vec, NULL, n);
    for (matx_int64_t i = 0; i < n; ++i)
    {
        is >> vec->data[i].real >> vec->data[i].imag;
    }

    is.close();

    ret = MATX_OK;
    return ret;
}
