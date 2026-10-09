#include "matx/matx_print.h"
#include "matx/matx_types_internal.h"

#include <fstream>
#include <iomanip>
#include <iostream>

void matx_print_dense_mtx_d_i8(const matx_dense_d_i8_t& mtx, const char* file)
{
    if (!mtx || !mtx->data)
        return;

    auto os = std::ofstream(file, std::ios::trunc);
    os << mtx->nrows << " " << mtx->ncols << " " << mtx->layout << std::endl;
    os << std::fixed << std::setprecision(15);

    for (matx_int64_t i = 0; i < mtx->nrows; ++i) {
        for (matx_int64_t j = 0; j < mtx->ncols; ++j) {
            auto idx = (mtx->layout == MATX_ROW_MAJOR) ? i * mtx->stride + j : j * mtx->stride + i;
            os << mtx->data[idx] << " ";
        }
        os << std::endl;
    }
    os.close();
}

void matx_print_dense_mtx_z_i8(const matx_dense_z_i8_t& mtx, const char* file)
{
    if (!mtx || !mtx->data)
        return;

    auto os = std::ofstream(file, std::ios::trunc);
    os << mtx->nrows << " " << mtx->ncols << " " << mtx->layout << std::endl;
    os << std::fixed << std::setprecision(15);

    for (matx_int64_t i = 0; i < mtx->nrows; ++i) {
        for (matx_int64_t j = 0; j < mtx->ncols; ++j) {
            auto idx = (mtx->layout == MATX_ROW_MAJOR) ? i * mtx->stride + j : j * mtx->stride + i;
            os << mtx->data[idx].real << " " << mtx->data[idx].imag << " ";
        }
        os << std::endl;
    }
    os.close();
}

void matx_print_sparse_mtx_d_i8(const matx_coo_d_i8_t& mtx, const char* file)
{
    if (!mtx || !mtx->values)
        return;

    auto os = std::ofstream(file, std::ios::trunc);
    os << mtx->nrows << " " << mtx->ncols << " " << mtx->nnz << std::endl;
    os << std::fixed << std::setprecision(15);

    for (matx_int64_t i = 0; i < mtx->nnz; ++i) {
        os << mtx->rows[i] << " " << mtx->columns[i] << " " << mtx->values[i] << std::endl;
    }
    os.close();
}

void matx_print_sparse_mtx_z_i8(const matx_coo_z_i8_t& mtx, const char* file)
{
    if (!mtx || !mtx->values)
        return;

    auto os = std::ofstream(file, std::ios::trunc);
    os << mtx->nrows << " " << mtx->ncols << " " << mtx->nnz << std::endl;
    os << std::fixed << std::setprecision(15);

    for (matx_int64_t i = 0; i < mtx->nnz; ++i) {
        os << mtx->rows[i] << " " << mtx->columns[i] << " " << mtx->values[i].real << " "
           << mtx->values[i].imag << std::endl;
    }
    os.close();
}

void matx_print_vec_d_i8(const matx_vec_d_i8_t& vec, const char* file)
{
    if (!vec || !vec->data)
        return;

    auto os = std::ofstream(file, std::ios::trunc);
    os << vec->n << std::endl;
    os << std::fixed << std::setprecision(15);

    for (matx_int64_t i = 0; i < vec->n; ++i) {
        os << vec->data[i * vec->stride] << std::endl;
    }
    os.close();
}

void matx_print_vec_z_i8(const matx_vec_z_i8_t& vec, const char* file)
{
    if (!vec || !vec->data)
        return;

    auto os = std::ofstream(file, std::ios::trunc);
    os << vec->n << std::endl;
    os << std::fixed << std::setprecision(15);
    for (matx_int64_t i = 0; i < vec->n; ++i) {
        os << vec->data[i * vec->stride].real << " " << vec->data[i * vec->stride].imag << std::endl;
    }
    os.close();
}

/* ============ Matrix info / summary ============ */

#include <cstdio>
#include <cstring>

int matx_dense_d_i8_info(const matx_dense_d_i8_t mtx, char* buf, size_t bufsize)
{
    if (!mtx || !buf || !bufsize)
        return -1;
    const char* layout_str = (mtx->layout == MATX_COL_MAJOR) ? "col-major" : "row-major";
    size_t data_bytes = (size_t) mtx->nrows * (size_t) mtx->ncols * sizeof(matx_double);
    double density = 100.0;
    int n = snprintf(buf, bufsize,
        "Dense[d]  rows=%lld  cols=%lld  layout=%s  stride=%lld  bytes=%zu  density=%.1f%%",
        (long long) mtx->nrows, (long long) mtx->ncols, layout_str,
        (long long) mtx->stride, data_bytes, density);
    if (n < 0) return -1;
    return n < (int) bufsize ? n : (int) bufsize - 1;
}

int matx_dense_z_i8_info(const matx_dense_z_i8_t mtx, char* buf, size_t bufsize)
{
    if (!mtx || !buf || !bufsize)
        return -1;
    const char* layout_str = (mtx->layout == MATX_COL_MAJOR) ? "col-major" : "row-major";
    size_t data_bytes = (size_t) mtx->nrows * (size_t) mtx->ncols * sizeof(matx_complex_d_t);
    int n = snprintf(buf, bufsize,
        "Dense[z]  rows=%lld  cols=%lld  layout=%s  stride=%lld  bytes=%zu",
        (long long) mtx->nrows, (long long) mtx->ncols, layout_str,
        (long long) mtx->stride, data_bytes);
    if (n < 0) return -1;
    return n < (int) bufsize ? n : (int) bufsize - 1;
}

int matx_coo_d_i8_info(const matx_coo_d_i8_t mtx, char* buf, size_t bufsize)
{
    if (!mtx || !buf || !bufsize)
        return -1;
    size_t idx_bytes = (size_t) mtx->nnz * 2 * sizeof(matx_int64_t);
    size_t val_bytes = (size_t) mtx->nnz * sizeof(matx_double);
    double density = (double) mtx->nnz / (double) ((size_t) mtx->nrows * (size_t) mtx->ncols) * 100.0;
    int n = snprintf(buf, bufsize,
        "COO[d]    rows=%lld  cols=%lld  nnz=%lld  idx_bytes=%zu  val_bytes=%zu  density=%.4f%%",
        (long long) mtx->nrows, (long long) mtx->ncols, (long long) mtx->nnz,
        idx_bytes, val_bytes, density);
    if (n < 0) return -1;
    return n < (int) bufsize ? n : (int) bufsize - 1;
}

int matx_coo_z_i8_info(const matx_coo_z_i8_t mtx, char* buf, size_t bufsize)
{
    if (!mtx || !buf || !bufsize)
        return -1;
    size_t idx_bytes = (size_t) mtx->nnz * 2 * sizeof(matx_int64_t);
    size_t val_bytes = (size_t) mtx->nnz * sizeof(matx_complex_d_t);
    double density = (double) mtx->nnz / (double) ((size_t) mtx->nrows * (size_t) mtx->ncols) * 100.0;
    int n = snprintf(buf, bufsize,
        "COO[z]    rows=%lld  cols=%lld  nnz=%lld  idx_bytes=%zu  val_bytes=%zu  density=%.4f%%",
        (long long) mtx->nrows, (long long) mtx->ncols, (long long) mtx->nnz,
        idx_bytes, val_bytes, density);
    if (n < 0) return -1;
    return n < (int) bufsize ? n : (int) bufsize - 1;
}
