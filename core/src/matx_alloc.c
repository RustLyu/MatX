#include <stdlib.h>

#include "matx/matx_func.h"
#include "matx/matx_types.h"

static void* matx_std_malloc(size_t size, void* user)
{
    (void) user;
    void* ptr = malloc(size);
    return ptr;
}

static void matx_std_free(void* ptr, void* user)
{
    (void) user;
    free(ptr);
}

matx_alloc_t matx_alloc_default(void)
{
    matx_alloc_t a;
    a.malloc_fn = &matx_std_malloc;
    a.free_fn = &matx_std_free;
    a.user = NULL;
    return a;
}

void* matx_malloc(const matx_alloc_t* a, size_t size)
{
    if (!a || !a->malloc_fn || !a->free_fn)
        return NULL;
    return a->malloc_fn(size, a->user);
}

void matx_free(const matx_alloc_t* a, void* ptr)
{
    if (!ptr || !a || !a->free_fn)
        return;
    a->free_fn(ptr, a->user);
}
