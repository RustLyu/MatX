#ifndef MATX_LOG_H
#define MATX_LOG_H

#include "matx/matx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MATX_LOG_TRACE = 0,
    MATX_LOG_DEBUG,
    MATX_LOG_INFO,
    MATX_LOG_WARN,
    MATX_LOG_ERROR,
    MATX_LOG_FATAL
} matx_log_level;

MATX_TOOLS_API void matx_log_init(const char* log_dir);

MATX_TOOLS_API void matx_log_set_level(matx_log_level level);

MATX_TOOLS_API void matx_log_trace(const char* file, int line, const char* fmt, ...);
MATX_TOOLS_API void matx_log_debug(const char* file, int line, const char* fmt, ...);
MATX_TOOLS_API void matx_log_info(const char* file, int line, const char* fmt, ...);
MATX_TOOLS_API void matx_log_warn(const char* file, int line, const char* fmt, ...);
MATX_TOOLS_API void matx_log_error(const char* file, int line, const char* fmt, ...);
MATX_TOOLS_API void matx_log_fatal(const char* file, int line, const char* fmt, ...);

#define MATX_TRACE(fmt, ...) matx_log_trace(__FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define MATX_DEBUG(fmt, ...) matx_log_debug(__FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define MATX_INFO(fmt, ...) matx_log_info(__FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define MATX_WARN(fmt, ...) matx_log_warn(__FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define MATX_ERROR(fmt, ...) matx_log_error(__FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define MATX_FATAL(fmt, ...) matx_log_fatal(__FILE__, __LINE__, fmt, ##__VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif