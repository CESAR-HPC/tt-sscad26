// SPDX-License-Identifier: Apache-2.0
//
// Tradução de um tipo C++ para o tt::DataFormat usado nos circular buffers.
#pragma once

#include <cstdint>
#include <type_traits>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace sscad {

template <typename T> constexpr tt::DataFormat data_format_of() {
    if constexpr (std::is_same_v<T, bfloat16>) {
        return tt::DataFormat::Float16_b;
    } else if constexpr (std::is_same_v<T, float>) {
        return tt::DataFormat::Float32;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        return tt::DataFormat::UInt32;
    } else {
        static_assert(sizeof(T) == 0, "Tipo sem DataFormat correspondente");
    }
}

// Tamanho em bytes de um tile 32x32 do tipo T (2048 B para bfloat16).
template <typename T> constexpr uint32_t tile_size_bytes() {
    return sizeof(T) * tt::constants::TILE_HW;
}

} // namespace sscad
