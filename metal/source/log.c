#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>

#define mindfour_ON_STACK_FORMAT_BUFFER_SIZE 16384

void mindfour_format_log(const char *format, va_list args)
{
    char stack_buffer[mindfour_ON_STACK_FORMAT_BUFFER_SIZE];
    char *heap_buffer = NULL;

    va_list args_copy;
    va_copy(args_copy, args);

    const int vsnprintf_result = vsnprintf(stack_buffer, mindfour_ON_STACK_FORMAT_BUFFER_SIZE, format, args);
    assert(vsnprintf_result >= 0);

    char *message_buffer = &stack_buffer[0];
    size_t message_size = (size_t)vsnprintf_result;
    if (message_size > mindfour_ON_STACK_FORMAT_BUFFER_SIZE)
    {
        heap_buffer = malloc(message_size);
        if (heap_buffer == NULL)
        {

            message_size = mindfour_ON_STACK_FORMAT_BUFFER_SIZE;
        }
        else
        {

            vsnprintf(heap_buffer, message_size, format, args_copy);
            message_buffer = heap_buffer;
        }
    }

    ssize_t bytes_written;
    do
    {
        bytes_written = write(STDERR_FILENO, message_buffer, message_size);
        if (bytes_written > 0)
        {
            assert((size_t)bytes_written <= message_size);
            message_buffer += bytes_written;
            message_size -= bytes_written;
        }
    } while (bytes_written >= 0 && message_size != 0);

cleanup:
    free(heap_buffer);
    va_end(args_copy);
}
