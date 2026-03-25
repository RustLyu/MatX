#include "matx/matx_log.h"

#include <spdlog/spdlog.h>
#include <spdlog/async.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>

#include <vector>
#include <memory>
#include <cstdarg>
#include <cstdio>

static std::shared_ptr<spdlog::logger> g_logger;

static void log_impl(spdlog::level::level_enum level,
    const char* file,
    int line,
    const char* fmt,
    va_list args)
{
    char buffer[2048];

    vsnprintf(buffer, sizeof(buffer), fmt, args);

    g_logger->log(spdlog::source_loc{ file, line, "" }, level, buffer);
}

void matx_log_init(const char* log_dir)
{
    std::string logfile = std::string(log_dir) + "/matx.log";

    auto console_sink =
        std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

    auto file_sink =
        std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            logfile,
            20 * 1024 * 1024,
            5);

    std::vector<spdlog::sink_ptr> sinks{ console_sink, file_sink };
    g_logger = std::make_shared<spdlog::logger>(
        "matx",
        sinks.begin(), sinks.end());

    g_logger->set_pattern(
        "%Y-%m-%d %H:%M:%S.%e UTC%z [%t] [%^%l%$] %s:%# | %v");

    g_logger->set_level(spdlog::level::trace);
    g_logger->flush_on(spdlog::level::trace);
    spdlog::flush_every(std::chrono::milliseconds(100));
    spdlog::set_default_logger(g_logger);
}

void matx_log_set_level(matx_log_level level)
{
    g_logger->set_level((spdlog::level::level_enum)level);
}

#define MATX_LOG_IMPL(name, lvl)                    \
void matx_log_##name(const char* file, int line,    \
                     const char* fmt, ...)          \
{                                                   \
    va_list args;                                   \
    va_start(args, fmt);                            \
    log_impl(lvl, file, line, fmt, args);           \
    va_end(args);                                   \
}

MATX_LOG_IMPL(trace, spdlog::level::trace)
MATX_LOG_IMPL(debug, spdlog::level::debug)
MATX_LOG_IMPL(info, spdlog::level::info)
MATX_LOG_IMPL(warn, spdlog::level::warn)
MATX_LOG_IMPL(error, spdlog::level::err)
MATX_LOG_IMPL(fatal, spdlog::level::critical)