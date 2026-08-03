#include "matx/matx_func.h"
#include "matx/matx_log.h"
#include "matx/matx_types.h"

const char* matx_status_string(matx_status_t st)
{
    switch (st) {
    case MATX_OK:
        return "MATX_OK";
    case MATX_ERR_INVALID_ARG:
        return "MATX_ERR_INVALID_ARG";
    case MATX_ERR_OUT_OF_MEMORY:
        return "MATX_ERR_OUT_OF_MEMORY";
    case MATX_ERR_NOT_SUPPORTED:
        return "MATX_ERR_NOT_SUPPORTED";
    case MATX_ERR_INTERNAL:
        return "MATX_ERR_INTERNAL";
    default:
        return "MATX_ERR_UNKNOWN";
    }
}
