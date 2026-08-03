#ifndef MATX_TYPES_INTERNAL_H
#define MATX_TYPES_INTERNAL_H

#include "matx/matx_types.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---- Backend handle helpers ----

static inline matx_handle_t* matx_handle_get(matx_handle_t** handles,
                                             int32_t* num,
                                             matx_handle_type_t type)
{
    for (int32_t i = 0; i < *num; ++i) {
        if ((*handles)[i].type == type)
            return &(*handles)[i];
    }
    int32_t new_num = *num + 1;
    matx_handle_t* new_handles
        = (matx_handle_t*) realloc(*handles, sizeof(matx_handle_t) * (size_t) new_num);
    if (!new_handles)
        return NULL;
    memset(&new_handles[*num], 0, sizeof(matx_handle_t));
    new_handles[*num].type = type;
    *handles = new_handles;
    *num = new_num;
    return &new_handles[*num - 1];
}

static inline void matx_handles_destroy(matx_handle_t** handles, int32_t* num)
{
    if (!handles || !*handles)
        return;
    for (int32_t i = 0; i < *num; ++i) {
        matx_handle_t* h = &(*handles)[i];
        if (h->valid > 0 && h->custom_free_func && h->impl) {
            h->custom_free_func(h->impl);
        }
    }
    free(*handles);
    *handles = NULL;
    *num = 0;
}

#define MATX_HANDLE(obj, type) matx_handle_get(&(obj)->backend_handles, &(obj)->num_backend_handles, type)

// ---- Dense vector (double) ----
typedef struct matx_vec_d_i8_opaque_t
{
    matx_int64_t n;
    matx_int64_t stride;
    matx_double* data;
    matx_int64_t flags;
    matx_handle_t* backend_handles;
    int32_t num_backend_handles;
} matx_vec_d_i8_opaque_t;

// ---- Dense matrix (double) ----
typedef struct matx_dense_d_i8_opaque_t
{
    matx_int64_t nrows;
    matx_int64_t ncols;
    matx_int64_t stride;
    // leading dimension: if col-major => ld = stride (>= rows);
    //                   if row-major => ld = stride (>= cols);
    matx_layout_t layout;
    matx_double* data;
    matx_int64_t flags; // reserved for future (ownership, alignment, etc.)
    matx_handle_t* backend_handles;
    int32_t num_backend_handles;
} matx_dense_d_i8_opaque_t;

// ---- Dense vector (complex) ----
typedef struct matx_vec_z_i8_opaque_t
{
    matx_int64_t n;
    matx_int64_t stride;
    matx_complex_d_t* data;
    matx_int64_t flags;
    matx_handle_t* backend_handles;
    int32_t num_backend_handles;
} matx_vec_z_i8_opaque_t;

// ---- Dense matrix (complex) ----
typedef struct matx_dense_z_i8_opaque_t
{
    matx_int64_t nrows;
    matx_int64_t ncols;
    matx_int64_t stride;
    matx_layout_t layout;
    matx_complex_d_t* data;
    matx_int64_t flags;
    matx_handle_t* backend_handles;
    int32_t num_backend_handles;
} matx_dense_z_i8_opaque_t;

// ---- Sparse CSC (real/complex) ----
typedef struct matx_csc_d_i8_opaque_t
{
    matx_int64_t nrows;
    matx_int64_t ncols;
    matx_int64_t nnz;
    matx_int64_t* col_ptr;
    matx_int64_t* row_ind;
    matx_double* values;
    matx_int64_t* coo_csc_index_map;
    matx_int64_t
        struct_update; // coo to csc conversion may involve sorting and duplicate summation, these flags can be used to track whether the structure/values are up to date with the original COO data
    matx_int64_t
        only_value_update; // if the structure is up to date, but values have been updated, this flag can be set to indicate that only values need to be updated in the CSC representation without redoing the entire COO to CSC conversion
    matx_int64_t flags;
} matx_csc_d_i8_opaque_t;

typedef struct matx_coo_d_i8_opaque_t
{
    matx_int64_t nrows;
    matx_int64_t ncols;
    matx_int64_t nnz;
    matx_int64_t* rows;
    matx_int64_t* columns;
    matx_double* values;
    matx_int64_t flags;
    matx_handle_t* backend_handles;
    int32_t num_backend_handles;
    matx_csc_d_i8_opaque_t*
        handle_csc; // for backends that require CSC format, we can lazily convert COO to CSC and store here to avoid repeated conversions
} matx_coo_d_i8_opaque_t;

typedef struct matx_csc_z_i8_opaque_t
{
    matx_int64_t nrows;
    matx_int64_t ncols;
    matx_int64_t nnz;
    matx_int64_t* col_ptr;
    matx_int64_t* row_ind;
    matx_complex_d_t* values;
    matx_int64_t* coo_csc_index_map;
    matx_int64_t
        struct_update; // coo to csc conversion may involve sorting and duplicate summation, these flags can be used to track whether the structure/values are up to date with the original COO data
    matx_int64_t
        only_value_update; // if the structure is up to date, but values have been updated, this flag can be set to indicate that only values need to be updated in the CSC representation without redoing the entire COO to CSC conversion
    matx_int64_t flags;
} matx_csc_z_i8_opaque_t;

typedef struct matx_coo_z_i8_opaque_t
{
    matx_int64_t nrows;
    matx_int64_t ncols;
    matx_int64_t nnz;
    matx_int64_t* rows;
    matx_int64_t* columns;
    matx_complex_d_t* values;
    matx_int64_t flags;
    matx_handle_t* backend_handles;
    int32_t num_backend_handles;
    matx_csc_z_i8_opaque_t*
        handle_csc; // for backends that require CSC format, we can lazily convert COO to CSC and store here to avoid repeated conversions
} matx_coo_z_i8_opaque_t;
#ifdef __cplusplus
}
#endif

#endif // MATX_TYPES_INTERNAL_H