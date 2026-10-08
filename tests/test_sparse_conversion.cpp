#include "matx_test_harness.h"

#include <limits>

extern "C" {
#include "matx/matx_sparse_solve.h"
#include "matx/matx_sparse_compute.h"
#include "matx/matx_types.h"
#include "matx/matx_types_internal.h"
}

typedef struct {
    matx_int64_t nrows;
    matx_int64_t ncols;
    matx_int64_t nnz;
    matx_int64_t* row_ptr;
    matx_int64_t* col_ind;
    void* val;
    size_t elem_size;
} matx_test_csr_matrix_t;

extern "C" int coo_to_csr_optimized(const matx_alloc_t* alloc,
                                     matx_int64_t nrows,
                                     matx_int64_t ncols,
                                     matx_int64_t nnz,
                                     const matx_int64_t* coo_row,
                                     const matx_int64_t* coo_col,
                                     const void* coo_val,
                                     size_t elem_size,
                                     matx_test_csr_matrix_t* csr);

TEST(sparse_solve, coo_to_csc_unsorted_duplicate_mapping)
{
    matx_alloc_t alloc = matx_alloc_default();
    matx_int64_t rows[] = {3, 1, 3, 1, 2, 0, 4, 3, 0};
    matx_int64_t columns[] = {4, 1, 4, 1, 0, 5, 2, 1, 5};
    matx_double values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const matx_int64_t expected_col_ptr[] = {0, 1, 3, 4, 4, 5, 6};
    const matx_int64_t expected_row_ind[] = {2, 1, 3, 4, 3, 0};
    const matx_int64_t expected_map[] = {4, 1, 4, 1, 0, 5, 3, 2, 5};

    matx_coo_d_i8_t matrix = nullptr;
    ASSERT_EQ(matx_coo_sparse_d_i8_create(&alloc, &matrix, 5, 6, 9,
                                           rows, columns, values), MATX_OK);
    ASSERT_EQ(coo_to_csc_d_i8(matrix), MATX_OK);
    ASSERT_NE(matrix->handle_csc, nullptr);
    EXPECT_EQ(matrix->handle_csc->nnz, 6);
    for (matx_int64_t i = 0; i < 7; ++i)
        EXPECT_EQ(matrix->handle_csc->col_ptr[i], expected_col_ptr[i]);
    for (matx_int64_t i = 0; i < 6; ++i)
        EXPECT_EQ(matrix->handle_csc->row_ind[i], expected_row_ind[i]);
    for (matx_int64_t i = 0; i < 9; ++i)
        EXPECT_EQ(matrix->handle_csc->coo_csc_index_map[i], expected_map[i]);
    matx_coo_sparse_d_i8_destroy(&alloc, matrix);
}

TEST(sparse_solve, coo_to_csc_sorted_duplicate_mapping_complex)
{
    matx_alloc_t alloc = matx_alloc_default();
    matx_int64_t rows[] = {0, 2, 2, 1, 3, 0};
    matx_int64_t columns[] = {0, 0, 0, 2, 2, 4};
    matx_complex_d_t values[] = {
        {1, 0}, {2, 1}, {3, -1}, {4, 0}, {5, 2}, {6, 0}};
    const matx_int64_t expected_col_ptr[] = {0, 2, 2, 4, 4, 5};
    const matx_int64_t expected_row_ind[] = {0, 2, 1, 3, 0};
    const matx_int64_t expected_map[] = {0, 1, 1, 2, 3, 4};

    matx_coo_z_i8_t matrix = nullptr;
    ASSERT_EQ(matx_coo_sparse_z_i8_create(&alloc, &matrix, 4, 5, 6,
                                           rows, columns, values), MATX_OK);
    ASSERT_EQ(coo_to_csc_z_i8(matrix), MATX_OK);
    ASSERT_NE(matrix->handle_csc, nullptr);
    EXPECT_EQ(matrix->handle_csc->nnz, 5);
    for (matx_int64_t i = 0; i < 6; ++i)
        EXPECT_EQ(matrix->handle_csc->col_ptr[i], expected_col_ptr[i]);
    for (matx_int64_t i = 0; i < 5; ++i)
        EXPECT_EQ(matrix->handle_csc->row_ind[i], expected_row_ind[i]);
    for (matx_int64_t i = 0; i < 6; ++i)
        EXPECT_EQ(matrix->handle_csc->coo_csc_index_map[i], expected_map[i]);
    matx_coo_sparse_z_i8_destroy(&alloc, matrix);
}

TEST(sparse_solve, coo_to_csc_large_column_introsort)
{
    matx_alloc_t alloc = matx_alloc_default();
    constexpr matx_int64_t count = 146;
    constexpr matx_int64_t unique_count = 73;
    matx_int64_t rows[count];
    matx_int64_t columns[count];
    matx_double values[count];
    for (matx_int64_t i = 0; i < count; ++i) {
        rows[i] = (i * 37) % unique_count;
        columns[i] = 1;
        values[i] = (matx_double) i;
    }

    matx_coo_d_i8_t matrix = nullptr;
    ASSERT_EQ(matx_coo_sparse_d_i8_create(&alloc, &matrix, unique_count, 3,
                                           count, rows, columns, values), MATX_OK);
    ASSERT_EQ(coo_to_csc_d_i8(matrix), MATX_OK);
    ASSERT_NE(matrix->handle_csc, nullptr);
    EXPECT_EQ(matrix->handle_csc->nnz, unique_count);
    EXPECT_EQ(matrix->handle_csc->col_ptr[0], 0);
    EXPECT_EQ(matrix->handle_csc->col_ptr[1], 0);
    EXPECT_EQ(matrix->handle_csc->col_ptr[2], unique_count);
    EXPECT_EQ(matrix->handle_csc->col_ptr[3], unique_count);
    for (matx_int64_t i = 0; i < unique_count; ++i)
        EXPECT_EQ(matrix->handle_csc->row_ind[i], i);
    for (matx_int64_t i = 0; i < count; ++i)
        EXPECT_EQ(matrix->handle_csc->coo_csc_index_map[i], rows[i]);
    matx_coo_sparse_d_i8_destroy(&alloc, matrix);
}

TEST(sparse_solve, coo_to_csc_late_unsorted_input)
{
    matx_alloc_t alloc = matx_alloc_default();
    constexpr matx_int64_t count = 40;
    matx_int64_t rows[count];
    matx_int64_t columns[count];
    matx_double values[count];
    for (matx_int64_t i = 0; i < 33; ++i) {
        rows[i] = i;
        columns[i] = i;
        values[i] = 1.0;
    }
    rows[33] = 33;
    columns[33] = 0;
    values[33] = 1.0;
    for (matx_int64_t i = 34; i < count; ++i) {
        rows[i] = i;
        columns[i] = i - 1;
        values[i] = 1.0;
    }

    matx_coo_d_i8_t matrix = nullptr;
    ASSERT_EQ(matx_coo_sparse_d_i8_create(&alloc, &matrix, count, count,
                                           count, rows, columns, values), MATX_OK);
    ASSERT_EQ(coo_to_csc_d_i8(matrix), MATX_OK);
    ASSERT_NE(matrix->handle_csc, nullptr);
    EXPECT_EQ(matrix->handle_csc->nnz, count);
    for (matx_int64_t col = 0; col < count; ++col)
        EXPECT_TRUE(matrix->handle_csc->col_ptr[col] <= matrix->handle_csc->col_ptr[col + 1]);
    for (matx_int64_t i = 0; i < count; ++i) {
        const matx_int64_t dst = matrix->handle_csc->coo_csc_index_map[i];
        EXPECT_TRUE(dst >= matrix->handle_csc->col_ptr[columns[i]]);
        EXPECT_TRUE(dst < matrix->handle_csc->col_ptr[columns[i] + 1]);
        EXPECT_EQ(matrix->handle_csc->row_ind[dst], rows[i]);
    }
    matx_coo_sparse_d_i8_destroy(&alloc, matrix);
}

TEST(sparse_solve, coo_to_csc_radix_sort_preserves_int64_rows_and_mapping)
{
    matx_alloc_t alloc = matx_alloc_default();
    constexpr matx_int64_t count = 300;
    const matx_int64_t large_row = std::numeric_limits<matx_int64_t>::max() - 1;
    matx_int64_t rows[count];
    matx_int64_t columns[count];
    matx_double values[count];
    const matx_int64_t row_pattern[] = {large_row, 7, 0, 1};
    for (matx_int64_t i = 0; i < count; ++i) {
        rows[i] = row_pattern[i % 4];
        columns[i] = 1;
        values[i] = 1.0;
    }

    matx_coo_d_i8_t matrix = nullptr;
    ASSERT_EQ(matx_coo_sparse_d_i8_create(&alloc, &matrix, large_row + 1,
                                           2, count, rows, columns, values), MATX_OK);
    ASSERT_EQ(coo_to_csc_d_i8(matrix), MATX_OK);
    ASSERT_NE(matrix->handle_csc, nullptr);
    EXPECT_EQ(matrix->handle_csc->nnz, 4);
    EXPECT_EQ(matrix->handle_csc->col_ptr[0], 0);
    EXPECT_EQ(matrix->handle_csc->col_ptr[1], 0);
    EXPECT_EQ(matrix->handle_csc->col_ptr[2], 4);
    EXPECT_EQ(matrix->handle_csc->row_ind[0], 0);
    EXPECT_EQ(matrix->handle_csc->row_ind[1], 1);
    EXPECT_EQ(matrix->handle_csc->row_ind[2], 7);
    EXPECT_EQ(matrix->handle_csc->row_ind[3], large_row);
    for (matx_int64_t i = 0; i < count; ++i) {
        matx_int64_t expected = 0;
        if (rows[i] == 1) expected = 1;
        if (rows[i] == 7) expected = 2;
        if (rows[i] == large_row) expected = 3;
        EXPECT_EQ(matrix->handle_csc->coo_csc_index_map[i], expected);
    }
    matx_coo_sparse_d_i8_destroy(&alloc, matrix);
}

TEST(sparse_blas, coo_to_dense_clears_contiguous_and_preserves_padding)
{
    matx_alloc_t alloc = matx_alloc_default();
    matx_int64_t rows[] = {0, 1, 1};
    matx_int64_t columns[] = {1, 0, 1};
    matx_double values[] = {2.0, 3.0, 4.0};
    matx_coo_d_i8_t real_coo = nullptr;
    ASSERT_EQ(matx_coo_sparse_d_i8_create(&alloc, &real_coo, 2, 2, 3,
                                           rows, columns, values), MATX_OK);
    double padded_data[] = {-99.0, -99.0, -99.0, -99.0, -99.0, -99.0};
    matx_dense_d_i8_t real_dense = nullptr;
    ASSERT_EQ(matx_dense_d_i8_wrap(&alloc, &real_dense, 2, 2, 3,
                                   MATX_ROW_MAJOR, padded_data), MATX_OK);
    ASSERT_EQ(matx_coo_to_dense_d_i8(real_coo, real_dense), MATX_OK);
    EXPECT_EQ(padded_data[0], 0.0);
    EXPECT_EQ(padded_data[1], 2.0);
    EXPECT_EQ(padded_data[2], -99.0);
    EXPECT_EQ(padded_data[3], 3.0);
    EXPECT_EQ(padded_data[4], 4.0);
    EXPECT_EQ(padded_data[5], -99.0);

    matx_int64_t complex_rows[] = {0, 1, 1};
    matx_int64_t complex_columns[] = {1, 0, 0};
    matx_complex_d_t complex_values[] = {{1.0, 2.0}, {3.0, 4.0}, {5.0, -2.0}};
    matx_coo_z_i8_t complex_coo = nullptr;
    ASSERT_EQ(matx_coo_sparse_z_i8_create(&alloc, &complex_coo, 2, 2, 3,
                                           complex_rows, complex_columns,
                                           complex_values), MATX_OK);
    matx_dense_z_i8_t complex_dense = nullptr;
    ASSERT_EQ(matx_dense_z_i8_create(&alloc, &complex_dense, MATX_COL_MAJOR,
                                     2, 2, nullptr), MATX_OK);
    for (int i = 0; i < 4; ++i)
        complex_dense->data[i] = {99.0, -99.0};
    ASSERT_EQ(matx_coo_to_dense_z_i8(complex_coo, complex_dense), MATX_OK);
    EXPECT_EQ(complex_dense->data[0].real, 0.0);
    EXPECT_EQ(complex_dense->data[0].imag, 0.0);
    EXPECT_EQ(complex_dense->data[1].real, 8.0);
    EXPECT_EQ(complex_dense->data[1].imag, 2.0);
    EXPECT_EQ(complex_dense->data[2].real, 1.0);
    EXPECT_EQ(complex_dense->data[2].imag, 2.0);
    EXPECT_EQ(complex_dense->data[3].real, 0.0);
    EXPECT_EQ(complex_dense->data[3].imag, 0.0);

    matx_dense_z_i8_destroy(&alloc, complex_dense);
    matx_coo_sparse_z_i8_destroy(&alloc, complex_coo);
    matx_dense_d_i8_destroy(&alloc, real_dense);
    matx_coo_sparse_d_i8_destroy(&alloc, real_coo);
}

TEST(sparse_blas, coo_to_csr_unsorted_duplicates)
{
    matx_alloc_t alloc = matx_alloc_default();
    const matx_int64_t rows[] = {3, 1, 3, 1, 2, 0, 4, 3, 0};
    const matx_int64_t columns[] = {4, 1, 4, 1, 0, 5, 2, 1, 5};
    const matx_double values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const matx_int64_t expected_row_ptr[] = {0, 1, 2, 3, 5, 6};
    const matx_int64_t expected_col_ind[] = {5, 1, 0, 1, 4, 2};
    const matx_double expected_values[] = {15, 6, 5, 8, 4, 7};

    matx_test_csr_matrix_t csr{};
    ASSERT_EQ(coo_to_csr_optimized(&alloc, 5, 6, 9, rows, columns, values,
                                    sizeof(matx_double), &csr), 0);
    EXPECT_EQ(csr.nnz, 6);
    matx_double* csr_vals = (matx_double*) csr.val;
    for (matx_int64_t i = 0; i < 6; ++i)
        EXPECT_EQ(csr.row_ptr[i], expected_row_ptr[i]);
    for (matx_int64_t i = 0; i < 6; ++i) {
        EXPECT_EQ(csr.col_ind[i], expected_col_ind[i]);
        EXPECT_NEAR(csr_vals[i], expected_values[i], 1e-15);
    }
    matx_free(&alloc, csr.row_ptr);
    matx_free(&alloc, csr.col_ind);
    matx_free(&alloc, csr.val);
}

TEST(sparse_blas, coo_to_csr_large_row_introsort)
{
    matx_alloc_t alloc = matx_alloc_default();
    constexpr matx_int64_t count = 146;
    constexpr matx_int64_t unique_count = 73;
    matx_int64_t rows[count];
    matx_int64_t columns[count];
    matx_double values[count];
    for (matx_int64_t i = 0; i < count; ++i) {
        rows[i] = 0;
        columns[i] = (i * 37) % unique_count;
        values[i] = (matx_double) (columns[i] + 1);
    }

    matx_test_csr_matrix_t csr{};
    ASSERT_EQ(coo_to_csr_optimized(&alloc, 1, unique_count, count,
                                    rows, columns, values, sizeof(matx_double),
                                    &csr), 0);
    EXPECT_EQ(csr.nnz, unique_count);
    matx_double* csr_vals = (matx_double*) csr.val;
    EXPECT_EQ(csr.row_ptr[0], 0);
    EXPECT_EQ(csr.row_ptr[1], unique_count);
    for (matx_int64_t col = 0; col < unique_count; ++col) {
        EXPECT_EQ(csr.col_ind[col], col);
        EXPECT_NEAR(csr_vals[col], (matx_double) (2 * (col + 1)), 1e-15);
    }
    matx_free(&alloc, csr.row_ptr);
    matx_free(&alloc, csr.col_ind);
    matx_free(&alloc, csr.val);
}

TEST(sparse_blas, coo_to_csr_rejects_out_of_range_indices)
{
    matx_alloc_t alloc = matx_alloc_default();
    const matx_int64_t valid_rows[] = {0, 1};
    const matx_int64_t invalid_rows[] = {0, 3};
    const matx_int64_t valid_columns[] = {0, 1};
    const matx_int64_t invalid_columns[] = {0, 2};
    const matx_double values[] = {1.0, 2.0};

    auto expect_invalid_and_clean = [&](const matx_int64_t* rows,
                                        const matx_int64_t* columns) {
        matx_test_csr_matrix_t csr{};
        EXPECT_NE(coo_to_csr_optimized(&alloc, 3, 2, 2, rows, columns, values,
                                             sizeof(matx_double), &csr), 0);
        EXPECT_EQ(csr.row_ptr, nullptr);
        EXPECT_EQ(csr.col_ind, nullptr);
        EXPECT_EQ(csr.val, nullptr);
        EXPECT_EQ(csr.nrows, 0);
        EXPECT_EQ(csr.ncols, 0);
    };

    expect_invalid_and_clean(invalid_rows, valid_columns);
    expect_invalid_and_clean(valid_rows, invalid_columns);
}
