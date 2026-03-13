#ifndef MATX_TM_H
#define MATX_TM_H

#include "matx/matx.h"

#ifdef __cplusplus
extern "C" {
#endif
    typedef enum
    {
        MATX_TM_SECOND = 0,
        MATX_TM_MILISECOND,
        MATX_TM_MICROSECOND,
        MATX_TM_NANOSECOND
    } matx_tm_unit;

	matx_int64_t matx_tm_now(matx_tm_unit unit);

#ifdef __cplusplus
}
#endif

#endif // MATX_TM_H