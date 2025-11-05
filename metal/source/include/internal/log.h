#pragma once

#include <stdarg.h>

void mindfour_format_log(const char *format, va_list args);

__attribute__((__format__(__printf__, 1, 2))) inline static void mindfour_log(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    mindfour_format_log(format, args);
    va_end(args);
}

#define mindfour_LOG_ERROR(message, ...) \
    mindfour_log("Error: " message "\n", ##__VA_ARGS__)

#define mindfour_LOG_WARNING(message, ...) \
    mindfour_log("Warning: " message "\n", ##__VA_ARGS__)
