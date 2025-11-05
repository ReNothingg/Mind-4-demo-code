#pragma once

#include <stdint.h>

inline static uint32_t rng_squares32(uint64_t offset, uint64_t seed)
{
    const uint64_t y = offset * seed;
    const uint64_t z = y + seed;

    /* Круг 1 */
    uint64_t x = y * y + y;
    x = (x >> 32) | (x << 32);

    /* Круг 2 */
    x = x * x + z;
    x = (x >> 32) | (x << 32);

    /* Круг 3 */
    x = x * x + y;
    x = (x >> 32) | (x << 32);

    /* Кругн 4 */
    x = x * x + z;
    return (uint32_t)(x >> 32);
}
