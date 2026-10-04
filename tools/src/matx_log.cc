#include "matx/matx_log.h"

#include <spdlog/async.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <vector>

static std::shared_ptr<spdlog::logger>
g_logger;

static void log_impl(
    spdlog::level::level_enum level, const char* file, int line, const char* fmt, va_list args)
{
    if (!g_logger || !fmt)
        return;
    char buffer[2048];

    vsnprintf(buffer, sizeof(buffer), fmt, args);

    g_logger->log(spdlog::source_loc{file, line, ""}, level, buffer);
}

void matx_log_init(const char* log_dir)
{
    const std::filesystem::path directory
        = (log_dir && *log_dir) ? std::filesystem::path(log_dir) : std::filesystem::path(".");

    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    std::vector<spdlog::sink_ptr> sinks{console_sink};

    std::error_code directory_error;
    std::filesystem::create_directories(directory, directory_error);
    if (!directory_error) {
        try {
            auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                (directory / "matx.log").string(), 20 * 1024 * 1024, 5);
            sinks.push_back(file_sink);
        } catch (const spdlog::spdlog_ex& e) {
            std::fprintf(stderr, "MatX: unable to open log file: %s\n", e.what());
        }
    } else {
        std::fprintf(stderr, "MatX: unable to create log directory '%s': %s\n",
                     directory.string().c_str(), directory_error.message().c_str());
    }
    g_logger = std::make_shared<spdlog::logger>
    ("matx", sinks.begin(), sinks.end());

    g_logger->set_pattern("%Y-%m-%d %H:%M:%S.%e UTC%z [%t] [%^%l%$] %s:%# | %v");

    g_logger->set_level(spdlog::level::trace);
    g_logger->flush_on(spdlog::level::err);
    spdlog::flush_every(std::chrono::milliseconds(100));
    spdlog::set_default_logger(g_logger);
}

void matx_log_set_level(matx_log_level level)
{
    if (g_logger)
        g_logger->set_level((spdlog::level::level_enum) level);
}

#define MATX_LOG_IMPL(name, lvl) \
    void matx_log_##name(const char* file, int line, const char* fmt, ...) \
    { \
        va_list args; \
        va_start(args, fmt); \
        log_impl(lvl, file, line, fmt, args); \
        va_end(args); \
    }

MATX_LOG_IMPL(trace, spdlog::level::trace)
MATX_LOG_IMPL(debug, spdlog::level::debug)
MATX_LOG_IMPL(info, spdlog::level::info)
MATX_LOG_IMPL(warn, spdlog::level::warn)
MATX_LOG_IMPL(error, spdlog::level::err)
MATX_LOG_IMPL(fatal, spdlog::level::critical)
