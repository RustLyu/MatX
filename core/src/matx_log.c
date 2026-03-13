#include <memory>

#include <spdlog/spdlog.h>
#include <spdlog/async.h>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>

void matx_log_init(const char* log_dir)
{

        // async thread pool
        spdlog::init_thread_pool(8192, 1);

        auto console_sink =
            std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

        auto file_sink =
            std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                logfile,
                20 * 1024 * 1024,
                5);

        std::vector<spdlog::sink_ptr> sinks{
            console_sink,
            file_sink
        };

        auto logger =
            std::make_shared<spdlog::async_logger>(
                "matx",
                sinks.begin(),
                sinks.end(),
                spdlog::thread_pool(),
                spdlog::async_overflow_policy::block
            );

        logger->set_pattern(
            "%Y-%m-%d %H:%M:%S.%e [t%t] [%^%l%$] %s:%# | %v"
        );

        logger->set_level(spdlog::level::info);

        spdlog::set_default_logger(logger);
}

void matx_log_set_level(int level)
{
    spdlog::set_level(static_cast<spdlog::level::level_enum>(level));
}