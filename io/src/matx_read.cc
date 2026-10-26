#include "matx/matx_read.h"
#include "matx/matx_func.h"
#include "matx/matx_log.h"
#include "matx/matx_types_internal.h"

#include <fstream>
#include <cstdlib>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

static bool checked_dense_size(matx_int64_t rows,
                               matx_int64_t cols,
                               size_t element_size,
                               size_t* count)
{
    if (rows <= 0 || cols <= 0 || !count
        || (uint64_t) rows > SIZE_MAX / (uint64_t) cols) {
        return false;
    }
    const size_t elements = (size_t) rows * (size_t) cols;
    if (elements > SIZE_MAX / element_size) {
        return false;
    }
    *count = elements;
    return true;
}

template <typename T, typename Reader>
static bool read_values(std::istream& stream,
                        size_t count,
                        std::vector<T>& values,
                        Reader reader)
{
    values.resize(count);
    for (size_t i = 0; i < count; ++i) {
        if (!reader(stream, values[i])) {
            return false;
        }
    }
    return true;
}

static bool read_complex_value(std::istream& stream, matx_complex_d_t& value)
{
    std::string token;
    if (!(stream >> token)) {
        return false;
    }

    char* real_end = nullptr;
    const double real = std::strtod(token.c_str(), &real_end);
    if (real_end == token.c_str()) {
        return false;
    }

    if (*real_end == '\0') {
        value.real = real;
        return static_cast<bool>(stream >> value.imag);
    }

    char* imag_end = nullptr;
    const double imag = std::strtod(real_end, &imag_end);
    if (imag_end == real_end || imag_end[0] != 'i' || imag_end[1] != '\0') {
        return false;
    }

    value.real = real;
    value.imag = imag;
    return true;
}

template <typename T, typename Handle, typename Create, typename Reader>
static matx_status_t read_dense(const matx_alloc_t* alloc,
                                Handle* out,
                                const char* file,
                                Create create,
                                Reader reader)
{
    if (!alloc || !alloc->malloc_fn || !alloc->free_fn || !out || !file) {
        return MATX_ERR_INVALID_ARG;
    }
    std::ifstream stream(file);
    if (!stream.is_open()) {
        MATX_ERROR("open file error. path:%s", file);
        return MATX_ERR_INTERNAL;
    }

    std::string first_line;
    if (!std::getline(stream, first_line)) {
        MATX_ERROR("empty dense matrix file. path:%s", file);
        return MATX_ERR_INVALID_ARG;
    }
    /* MatrixMarket files carry a "%%MatrixMarket ..." banner line; the legacy
     * print format starts directly with "rows cols layout". */
    const bool is_mtx = first_line.rfind("%%MatrixMarket", 0) == 0;

    matx_int64_t rows = 0, cols = 0;
    matx_layout_t layout = MATX_COL_MAJOR;
    std::string dim_line;
    if (is_mtx) {
        if (!std::getline(stream, dim_line)) {
            MATX_ERROR("invalid dense matrix header. path:%s", file);
            return MATX_ERR_INVALID_ARG;
        }
        std::istringstream dims(dim_line);
        if (!(dims >> rows >> cols) || rows <= 0 || cols <= 0) {
            MATX_ERROR("invalid dense matrix header. path:%s", file);
            return MATX_ERR_INVALID_ARG;
        }
    } else {
        matx_int64_t layout_value = 0;
        std::istringstream dims(first_line);
        if (!(dims >> rows >> cols >> layout_value)
            || (layout_value != MATX_ROW_MAJOR && layout_value != MATX_COL_MAJOR)) {
            MATX_ERROR("invalid dense matrix header. path:%s", file);
            return MATX_ERR_INVALID_ARG;
        }
        layout = static_cast<matx_layout_t>(layout_value);
    }
    size_t count = 0;
    if (!checked_dense_size(rows, cols, sizeof(T), &count)) {
        MATX_ERROR("invalid dense matrix dimensions. path:%s", file);
        return MATX_ERR_INVALID_ARG;
    }

    try {
        std::vector<T> values;
        if (!read_values(stream, count, values, reader)) {
            MATX_ERROR("incomplete dense matrix data. path:%s", file);
            return MATX_ERR_INVALID_ARG;
        }
        /* MatrixMarket array format is already column-major on disk. */
        if (is_mtx) {
            return create(alloc, out, MATX_COL_MAJOR, rows, cols, values.data());
        }
        /* Legacy print format stores values in row-major order. */
        if (layout == MATX_ROW_MAJOR) {
            return create(alloc, out, layout, rows, cols, values.data());
        }
        std::vector<T> column_major(count);
        for (matx_int64_t r = 0; r < rows; ++r) {
            for (matx_int64_t c = 0; c < cols; ++c) {
                column_major[(size_t) c * (size_t) rows + (size_t) r]
                    = values[(size_t) r * (size_t) cols + (size_t) c];
            }
        }
        return create(alloc, out, layout, rows, cols, column_major.data());
    } catch (const std::bad_alloc&) {
        MATX_ERROR("out of memory reading dense matrix. path:%s", file);
        return MATX_ERR_OUT_OF_MEMORY;
    } catch (const std::length_error&) {
        MATX_ERROR("dense matrix is too large. path:%s", file);
        return MATX_ERR_INVALID_ARG;
    }
}

template <typename T, typename Handle, typename Create, typename Reader>
static matx_status_t read_sparse(const matx_alloc_t* alloc,
                                 Handle* out,
                                 const char* file,
                                 Create create,
                                 Reader reader)
{
    if (!alloc || !alloc->malloc_fn || !alloc->free_fn || !out || !file) {
        return MATX_ERR_INVALID_ARG;
    }
    std::ifstream stream(file);
    if (!stream.is_open()) {
        MATX_ERROR("open file error. path:%s", file);
        return MATX_ERR_INTERNAL;
    }

    std::string first_line;
    if (!std::getline(stream, first_line)) {
        MATX_ERROR("empty sparse matrix file. path:%s", file);
        return MATX_ERR_INVALID_ARG;
    }
    /* MatrixMarket coordinate files use 1-based indices; the legacy print
     * format uses 0-based indices with no banner line. */
    const bool is_mtx = first_line.rfind("%%MatrixMarket", 0) == 0;

    matx_int64_t rows = 0, cols = 0, nnz = 0;
    std::string dim_line;
    if (is_mtx) {
        if (!std::getline(stream, dim_line)) {
            MATX_ERROR("invalid sparse matrix header. path:%s", file);
            return MATX_ERR_INVALID_ARG;
        }
    } else {
        dim_line = first_line;
    }
    std::istringstream dims(dim_line);
    if (!(dims >> rows >> cols >> nnz) || rows <= 0 || cols <= 0 || nnz <= 0
        || (uint64_t) nnz > SIZE_MAX / sizeof(matx_int64_t)
        || (uint64_t) nnz > SIZE_MAX / sizeof(T)) {
        MATX_ERROR("invalid sparse matrix header. path:%s", file);
        return MATX_ERR_INVALID_ARG;
    }

    try {
        std::vector<matx_int64_t> row_indices((size_t) nnz);
        std::vector<matx_int64_t> col_indices((size_t) nnz);
        std::vector<T> values((size_t) nnz);
        for (size_t i = 0; i < (size_t) nnz; ++i) {
            matx_int64_t r = 0, c = 0;
            if (!(stream >> r >> c)
                || !reader(stream, values[i])) {
                MATX_ERROR("incomplete sparse matrix data. path:%s", file);
                return MATX_ERR_INVALID_ARG;
            }
            if (is_mtx) {
                --r;
                --c;
            }
            row_indices[i] = r;
            col_indices[i] = c;
        }
        return create(alloc, out, rows, cols, nnz, row_indices.data(),
                      col_indices.data(), values.data());
    } catch (const std::bad_alloc&) {
        MATX_ERROR("out of memory reading sparse matrix. path:%s", file);
        return MATX_ERR_OUT_OF_MEMORY;
    } catch (const std::length_error&) {
        MATX_ERROR("sparse matrix is too large. path:%s", file);
        return MATX_ERR_INVALID_ARG;
    }
}

template <typename T, typename Handle, typename Create, typename Reader>
static matx_status_t read_vector(const matx_alloc_t* alloc,
                                 Handle* out,
                                 const char* file,
                                 Create create,
                                 Reader reader)
{
    if (!alloc || !alloc->malloc_fn || !alloc->free_fn || !out || !file) {
        return MATX_ERR_INVALID_ARG;
    }
    std::ifstream stream(file);
    if (!stream.is_open()) {
        MATX_ERROR("open file error. path:%s", file);
        return MATX_ERR_INTERNAL;
    }

    std::string first_line;
    if (!std::getline(stream, first_line)) {
        MATX_ERROR("empty vector file. path:%s", file);
        return MATX_ERR_INVALID_ARG;
    }
    const bool is_mtx = first_line.rfind("%%MatrixMarket", 0) == 0;

    matx_int64_t n = 0;
    std::string dim_line;
    if (is_mtx) {
        if (!std::getline(stream, dim_line)) {
            MATX_ERROR("invalid vector header. path:%s", file);
            return MATX_ERR_INVALID_ARG;
        }
        std::istringstream dims(dim_line);
        matx_int64_t cols = 0;
        if (!(dims >> n >> cols) || n <= 0 || cols != 1) {
            MATX_ERROR("invalid vector header. path:%s", file);
            return MATX_ERR_INVALID_ARG;
        }
    } else {
        std::istringstream dims(first_line);
        if (!(dims >> n) || n <= 0) {
            MATX_ERROR("invalid vector header. path:%s", file);
            return MATX_ERR_INVALID_ARG;
        }
    }
    if ((uint64_t) n > SIZE_MAX / sizeof(T)) {
        MATX_ERROR("invalid vector header. path:%s", file);
        return MATX_ERR_INVALID_ARG;
    }
    try {
        std::vector<T> values;
        if (!read_values(stream, (size_t) n, values, reader)) {
            MATX_ERROR("incomplete vector data. path:%s", file);
            return MATX_ERR_INVALID_ARG;
        }
        return create(alloc, out, values.data(), n);
    } catch (const std::bad_alloc&) {
        MATX_ERROR("out of memory reading vector. path:%s", file);
        return MATX_ERR_OUT_OF_MEMORY;
    } catch (const std::length_error&) {
        MATX_ERROR("vector is too large. path:%s", file);
        return MATX_ERR_INVALID_ARG;
    }
}

} // namespace

matx_status_t matx_read_dense_mtx_d_i8(const matx_alloc_t* alloc,
                                       matx_dense_d_i8_t* mtx,
                                       const char* file)
{
    return read_dense<matx_double>(
        alloc, mtx, file,
        [](const matx_alloc_t* a, matx_dense_d_i8_t* out, matx_layout_t layout,
           matx_int64_t rows, matx_int64_t cols, matx_double* data) {
            return matx_dense_d_i8_create(a, out, layout, rows, cols, data);
        },
        [](std::istream& input, matx_double& value) {
            return static_cast<bool>(input >> value);
        });
}

matx_status_t matx_read_dense_mtx_z_i8(const matx_alloc_t* alloc,
                                       matx_dense_z_i8_t* mtx,
                                       const char* file)
{
    return read_dense<matx_complex_d_t>(
        alloc, mtx, file,
        [](const matx_alloc_t* a, matx_dense_z_i8_t* out, matx_layout_t layout,
           matx_int64_t rows, matx_int64_t cols, matx_complex_d_t* data) {
            return matx_dense_z_i8_create(a, out, layout, rows, cols, data);
        },
        read_complex_value);
}

matx_status_t matx_read_sparse_mtx_d_i8(const matx_alloc_t* alloc,
                                        matx_coo_d_i8_t* mtx,
                                        const char* file)
{
    return read_sparse<matx_double>(
        alloc, mtx, file,
        [](const matx_alloc_t* a, matx_coo_d_i8_t* out, matx_int64_t rows,
           matx_int64_t cols, matx_int64_t nnz, matx_int64_t* row_indices,
           matx_int64_t* col_indices, matx_double* values) {
            return matx_coo_sparse_d_i8_create(a, out, rows, cols, nnz,
                                               row_indices, col_indices, values);
        },
        [](std::istream& input, matx_double& value) {
            return static_cast<bool>(input >> value);
        });
}

matx_status_t matx_read_sparse_mtx_z_i8(const matx_alloc_t* alloc,
                                        matx_coo_z_i8_t* mtx,
                                        const char* file)
{
    return read_sparse<matx_complex_d_t>(
        alloc, mtx, file,
        [](const matx_alloc_t* a, matx_coo_z_i8_t* out, matx_int64_t rows,
           matx_int64_t cols, matx_int64_t nnz, matx_int64_t* row_indices,
           matx_int64_t* col_indices, matx_complex_d_t* values) {
            return matx_coo_sparse_z_i8_create(a, out, rows, cols, nnz,
                                               row_indices, col_indices, values);
        },
        read_complex_value);
}

matx_status_t matx_read_vec_d_i8(const matx_alloc_t* alloc,
                                 matx_vec_d_i8_t* vec,
                                 const char* file)
{
    return read_vector<matx_double>(
        alloc, vec, file,
        [](const matx_alloc_t* a, matx_vec_d_i8_t* out, matx_double* data,
           matx_int64_t n) { return matx_vec_d_i8_create(a, out, data, n); },
        [](std::istream& input, matx_double& value) {
            return static_cast<bool>(input >> value);
        });
}

matx_status_t matx_read_vec_z_i8(const matx_alloc_t* alloc,
                                 matx_vec_z_i8_t* vec,
                                 const char* file)
{
    return read_vector<matx_complex_d_t>(
        alloc, vec, file,
        [](const matx_alloc_t* a, matx_vec_z_i8_t* out,
           matx_complex_d_t* data, matx_int64_t n) {
            return matx_vec_z_i8_create(a, out, data, n);
        },
        read_complex_value);
}

/* ============ CSV read ============ */

namespace {

static bool parse_csv_line(const std::string& line,
                           std::vector<double>& values)
{
    values.clear();
    std::string token;
    bool in_quotes = false;
    for (size_t i = 0; i <= line.size(); ++i) {
        char c = (i < line.size()) ? line[i] : ',';
        if (c == '"') {
            in_quotes = !in_quotes;
            continue;
        }
        if (!in_quotes && c == ',') {
            if (!token.empty()) {
                char* end = nullptr;
                double v = std::strtod(token.c_str(), &end);
                if (end == token.c_str()) {
                    return false;
                }
                values.push_back(v);
                token.clear();
            } else {
                values.push_back(0.0);
            }
        } else if (!in_quotes && (c == '\r' || c == '\n')) {
            if (!token.empty()) {
                char* end = nullptr;
                double v = std::strtod(token.c_str(), &end);
                if (end == token.c_str()) {
                    return false;
                }
                values.push_back(v);
            }
            return true;
        } else {
            token += c;
        }
    }
    if (!token.empty()) {
        char* end = nullptr;
        double v = std::strtod(token.c_str(), &end);
        if (end == token.c_str()) {
            return false;
        }
        values.push_back(v);
    }
    return true;
}

static void set_values(std::vector<matx_double>& dst,
                       const std::vector<std::vector<double>>& src,
                       matx_int64_t nrows, matx_int64_t ncols)
{
    for (matx_int64_t r = 0; r < nrows; ++r) {
        for (matx_int64_t c = 0; c < ncols; ++c) {
            dst[(size_t) c * (size_t) nrows + (size_t) r] = src[(size_t) r][(size_t) c];
        }
    }
}

static void set_values(std::vector<matx_complex_d_t>& dst,
                       const std::vector<std::vector<double>>& src,
                       matx_int64_t nrows, matx_int64_t ncols)
{
    /* each matrix element occupies two raw double columns (real, imag) */
    for (matx_int64_t r = 0; r < nrows; ++r) {
        for (matx_int64_t c = 0; c < ncols; ++c) {
            matx_complex_d_t& val = dst[(size_t) c * (size_t) nrows + (size_t) r];
            val.real = src[(size_t) r][(size_t) c * 2];
            val.imag = src[(size_t) r][(size_t) c * 2 + 1];
        }
    }
}

template <typename T, typename CreateFn>
static matx_status_t read_csv_impl(const matx_alloc_t* alloc,
                                   const char* file,
                                   CreateFn create)
{
    if (!alloc || !alloc->malloc_fn || !alloc->free_fn || !file) {
        return MATX_ERR_INVALID_ARG;
    }
    std::ifstream stream(file);
    if (!stream.is_open()) {
        MATX_ERROR("open file error. path:%s", file);
        return MATX_ERR_INTERNAL;
    }
    try {
        std::vector<std::vector<double>> all_rows;
        std::string line;
        matx_int64_t ncols = 0;
        while (std::getline(stream, line)) {
            if (line.empty())
                continue;
            std::vector<double> row_values;
            if (!parse_csv_line(line, row_values)) {
                MATX_ERROR("invalid CSV data. path:%s", file);
                return MATX_ERR_INVALID_ARG;
            }
            if (row_values.empty())
                continue;
            if (ncols == 0)
                ncols = (matx_int64_t) row_values.size();
            else if ((matx_int64_t) row_values.size() != ncols) {
                MATX_ERROR("inconsistent CSV column count. path:%s", file);
                return MATX_ERR_INVALID_ARG;
            }
            all_rows.push_back(std::move(row_values));
        }
        if (all_rows.empty() || ncols == 0) {
            MATX_ERROR("empty CSV file. path:%s", file);
            return MATX_ERR_INVALID_ARG;
        }
        matx_int64_t nrows = (matx_int64_t) all_rows.size();
        /* a complex element spans two raw double columns, so recover the
         * true matrix column count before sizing the destination. */
        const matx_int64_t cols_factor = (matx_int64_t)(sizeof(T) / sizeof(double));
        if (ncols % cols_factor != 0) {
            MATX_ERROR("inconsistent CSV column count. path:%s", file);
            return MATX_ERR_INVALID_ARG;
        }
        const matx_int64_t matrix_cols = ncols / cols_factor;
        size_t count = (size_t) nrows * (size_t) matrix_cols;
        if (count > SIZE_MAX / sizeof(T)) {
            MATX_ERROR("CSV data too large. path:%s", file);
            return MATX_ERR_INVALID_ARG;
        }
        std::vector<T> col_major(count);
        set_values(col_major, all_rows, nrows, matrix_cols);
        return create(alloc, nrows, matrix_cols, col_major.data());
    } catch (const std::bad_alloc&) {
        MATX_ERROR("out of memory reading CSV. path:%s", file);
        return MATX_ERR_OUT_OF_MEMORY;
    }
}

} // namespace

matx_status_t matx_read_csv_d_i8(const matx_alloc_t* alloc,
                                 matx_dense_d_i8_t* mtx,
                                 const char* file)
{
    if (!mtx) {
        return MATX_ERR_INVALID_ARG;
    }
    return read_csv_impl<matx_double>(
        alloc, file,
        [mtx](const matx_alloc_t* a, matx_int64_t rows, matx_int64_t cols, matx_double* data) {
            return matx_dense_d_i8_create(a, mtx, MATX_COL_MAJOR, rows, cols, data);
        });
}

matx_status_t matx_read_csv_z_i8(const matx_alloc_t* alloc,
                                 matx_dense_z_i8_t* mtx,
                                 const char* file)
{
    if (!mtx) {
        return MATX_ERR_INVALID_ARG;
    }
    return read_csv_impl<matx_complex_d_t>(
        alloc, file,
        [mtx](const matx_alloc_t* a, matx_int64_t rows, matx_int64_t cols, matx_complex_d_t* data) {
            return matx_dense_z_i8_create(a, mtx, MATX_COL_MAJOR, rows, cols, data);
        });
}

/* ---- CSV vector read ---- */

matx_status_t matx_read_csv_vec_d_i8(const matx_alloc_t* alloc,
                                      matx_vec_d_i8_t* vec,
                                      const char* file)
{
    if (!alloc || !alloc->malloc_fn || !alloc->free_fn || !vec || !file) {
        return MATX_ERR_INVALID_ARG;
    }
    std::ifstream stream(file);
    if (!stream.is_open()) {
        MATX_ERROR("open file error. path:%s", file);
        return MATX_ERR_INTERNAL;
    }
    try {
        std::vector<double> values;
        std::string line;
        while (std::getline(stream, line)) {
            if (line.empty())
                continue;
            char* end = nullptr;
            double v = std::strtod(line.c_str(), &end);
            if (end == line.c_str()) {
                MATX_ERROR("invalid CSV data. path:%s", file);
                return MATX_ERR_INVALID_ARG;
            }
            values.push_back(v);
        }
        if (values.empty()) {
            MATX_ERROR("empty CSV file. path:%s", file);
            return MATX_ERR_INVALID_ARG;
        }
        return matx_vec_d_i8_create(alloc, vec, values.data(),
                                     (matx_int64_t) values.size());
    } catch (const std::bad_alloc&) {
        MATX_ERROR("out of memory reading CSV vector. path:%s", file);
        return MATX_ERR_OUT_OF_MEMORY;
    }
}

matx_status_t matx_read_csv_vec_z_i8(const matx_alloc_t* alloc,
                                      matx_vec_z_i8_t* vec,
                                      const char* file)
{
    if (!alloc || !alloc->malloc_fn || !alloc->free_fn || !vec || !file) {
        return MATX_ERR_INVALID_ARG;
    }
    std::ifstream stream(file);
    if (!stream.is_open()) {
        MATX_ERROR("open file error. path:%s", file);
        return MATX_ERR_INTERNAL;
    }
    try {
        std::vector<matx_complex_d_t> values;
        std::string line;
        while (std::getline(stream, line)) {
            if (line.empty())
                continue;
            matx_complex_d_t v = {0.0, 0.0};
            char* end = nullptr;
            v.real = std::strtod(line.c_str(), &end);
            if (end == line.c_str()) {
                MATX_ERROR("invalid CSV data. path:%s", file);
                return MATX_ERR_INVALID_ARG;
            }
            /* skip comma and optional whitespace */
            while (*end == ',' || *end == ' ' || *end == '\t')
                ++end;
            if (*end != '\0' && *end != '\r' && *end != '\n') {
                char* imag_end = nullptr;
                v.imag = std::strtod(end, &imag_end);
            }
            values.push_back(v);
        }
        if (values.empty()) {
            MATX_ERROR("empty CSV file. path:%s", file);
            return MATX_ERR_INVALID_ARG;
        }
        return matx_vec_z_i8_create(alloc, vec, values.data(),
                                     (matx_int64_t) values.size());
    } catch (const std::bad_alloc&) {
        MATX_ERROR("out of memory reading CSV vector. path:%s", file);
        return MATX_ERR_OUT_OF_MEMORY;
    }
}
