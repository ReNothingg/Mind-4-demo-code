#pragma once

#include <cstdint>

namespace mindfour
{

    namespace rng
    {

        inline static std::uint32_t squares32(std::uint64_t offset, std::uint64_t seed)
        {
            const std::uint64_t y = offset * seed;
            const std::uint64_t z = y + seed;

            /* Кругн 1 */
            std::uint64_t x = y * y + y;
            x = (x >> 32) | (x << 32);

            /* Круг 2 */
            x = x * x + z;
            x = (x >> 32) | (x << 32);

            /* Круг 3 */
            x = x * x + y;
            x = (x >> 32) | (x << 32);

            /* Круг 4 */
            x = x * x + z;
            return static_cast<uint32_t>(x >> 32);
        }

    } // namespace rng

} // namespace mindfour
