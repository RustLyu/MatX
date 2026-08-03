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
            os << mtx->data[idx].real << "+" << mtx->data[idx].imag << "i ";
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
        os << vec->data[i] << std::endl;
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
        os << vec->data[i].real << " " << vec->data[i].imag << std::endl;
    }
    os.close();
}
