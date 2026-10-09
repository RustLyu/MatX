#include "matx/matx_write.h"
#include "matx/matx_types_internal.h"

#include <fstream>
#include <iomanip>
#include <cstdio>

/* ============ Standard Matrix Market (.mtx) format output ============ */

void matx_write_dense_mtx_d_i8(const matx_dense_d_i8_t mtx, const char* file)
{
    if (!mtx || !mtx->data || !file)
        return;
    std::ofstream os(file, std::ios::trunc);
    os << "%%MatrixMarket matrix array real general\n";
    os << mtx->nrows << " " << mtx->ncols << "\n";
    os << std::scientific << std::setprecision(15);
    for (matx_int64_t j = 0; j < mtx->ncols; ++j) {
        for (matx_int64_t i = 0; i < mtx->nrows; ++i) {
            matx_int64_t idx = (mtx->layout == MATX_ROW_MAJOR)
                ? i * mtx->stride + j : i + j * mtx->stride;
            os << mtx->data[idx] << "\n";
        }
    }
}

void matx_write_dense_mtx_z_i8(const matx_dense_z_i8_t mtx, const char* file)
{
    if (!mtx || !mtx->data || !file)
        return;
    std::ofstream os(file, std::ios::trunc);
    os << "%%MatrixMarket matrix array complex general\n";
    os << mtx->nrows << " " << mtx->ncols << "\n";
    os << std::scientific << std::setprecision(15);
    for (matx_int64_t j = 0; j < mtx->ncols; ++j) {
        for (matx_int64_t i = 0; i < mtx->nrows; ++i) {
            matx_int64_t idx = (mtx->layout == MATX_ROW_MAJOR)
                ? i * mtx->stride + j : i + j * mtx->stride;
            os << mtx->data[idx].real << " " << mtx->data[idx].imag << "\n";
        }
    }
}

void matx_write_sparse_mtx_d_i8(const matx_coo_d_i8_t mtx, const char* file)
{
    if (!mtx || !mtx->values || !file)
        return;
    std::ofstream os(file, std::ios::trunc);
    os << "%%MatrixMarket matrix coordinate real general\n";
    os << mtx->nrows << " " << mtx->ncols << " " << mtx->nnz << "\n";
    os << std::scientific << std::setprecision(15);
    for (matx_int64_t k = 0; k < mtx->nnz; ++k)
        os << (mtx->rows[k] + 1) << " " << (mtx->columns[k] + 1) << " "
           << mtx->values[k] << "\n";
}

void matx_write_sparse_mtx_z_i8(const matx_coo_z_i8_t mtx, const char* file)
{
    if (!mtx || !mtx->values || !file)
        return;
    std::ofstream os(file, std::ios::trunc);
    os << "%%MatrixMarket matrix coordinate complex general\n";
    os << mtx->nrows << " " << mtx->ncols << " " << mtx->nnz << "\n";
    os << std::scientific << std::setprecision(15);
    for (matx_int64_t k = 0; k < mtx->nnz; ++k)
        os << (mtx->rows[k] + 1) << " " << (mtx->columns[k] + 1) << " "
           << mtx->values[k].real << " " << mtx->values[k].imag << "\n";
}

void matx_write_vec_d_i8(const matx_vec_d_i8_t vec, const char* file)
{
    if (!vec || !vec->data || !file)
        return;
    std::ofstream os(file, std::ios::trunc);
    os << "%%MatrixMarket matrix array real general\n";
    os << vec->n << " 1\n";
    os << std::scientific << std::setprecision(15);
    for (matx_int64_t i = 0; i < vec->n; ++i)
        os << vec->data[i * vec->stride] << "\n";
}

void matx_write_vec_z_i8(const matx_vec_z_i8_t vec, const char* file)
{
    if (!vec || !vec->data || !file)
        return;
    std::ofstream os(file, std::ios::trunc);
    os << "%%MatrixMarket matrix array complex general\n";
    os << vec->n << " 1\n";
    os << std::scientific << std::setprecision(15);
    for (matx_int64_t i = 0; i < vec->n; ++i)
        os << vec->data[i * vec->stride].real << " "
           << vec->data[i * vec->stride].imag << "\n";
}

/* ============ CSV format output ============ */

void matx_write_csv_d_i8(const matx_dense_d_i8_t mtx, const char* file)
{
    if (!mtx || !mtx->data || !file)
        return;
    std::ofstream os(file, std::ios::trunc);
    os << std::scientific << std::setprecision(15);
    for (matx_int64_t i = 0; i < mtx->nrows; ++i) {
        for (matx_int64_t j = 0; j < mtx->ncols; ++j) {
            matx_int64_t idx = (mtx->layout == MATX_ROW_MAJOR)
                ? i * mtx->stride + j : i + j * mtx->stride;
            os << mtx->data[idx];
            if (j + 1 < mtx->ncols)
                os << ",";
        }
        os << "\n";
    }
}

void matx_write_csv_z_i8(const matx_dense_z_i8_t mtx, const char* file)
{
    if (!mtx || !mtx->data || !file)
        return;
    std::ofstream os(file, std::ios::trunc);
    os << std::scientific << std::setprecision(15);
    for (matx_int64_t i = 0; i < mtx->nrows; ++i) {
        for (matx_int64_t j = 0; j < mtx->ncols; ++j) {
            matx_int64_t idx = (mtx->layout == MATX_ROW_MAJOR)
                ? i * mtx->stride + j : i + j * mtx->stride;
            os << mtx->data[idx].real << "," << mtx->data[idx].imag;
            if (j + 1 < mtx->ncols)
                os << ",";
        }
        os << "\n";
    }
}