// SPDX-License-Identifier: Apache-2.0
//
// Conversão entre o layout row-major do host e o layout em tiles do Tensix.
//
// No dispositivo, uma matriz é armazenada como uma sequência de tiles 32x32
// (em ordem row-major de tiles). Cada tile, por sua vez, é dividido em 4 faces
// 16x16, também em ordem row-major. As unidades de cálculo do Tensix esperam
// exatamente esse formato, por isso o host converte os dados antes de
// enviá-los e desconverte o resultado ao recebê-lo.
#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tilize_utils.hpp>

namespace sscad {

constexpr uint32_t TILE = tt::constants::TILE_HEIGHT; // 32

// Arredonda n para o próximo múltiplo de 32.
constexpr uint32_t round_up_to_tile(uint32_t n) { return (n + TILE - 1) / TILE * TILE; }

// Copia uma matriz rows x cols para uma matriz maior (múltipla de 32),
// preenchendo o restante com zeros.
template <typename T>
std::vector<T> pad_to_tiles(const std::vector<T> &a, uint32_t rows, uint32_t cols) {
    const uint32_t prows = round_up_to_tile(rows);
    const uint32_t pcols = round_up_to_tile(cols);
    std::vector<T> out(static_cast<size_t>(prows) * pcols, T(0));
    for (uint32_t i = 0; i < rows; i++) {
        for (uint32_t j = 0; j < cols; j++) {
            out[i * pcols + j] = a[i * cols + j];
        }
    }
    return out;
}

// Operação inversa de pad_to_tiles: descarta as linhas e colunas de preenchimento.
template <typename T>
std::vector<T> unpad(const std::vector<T> &a, uint32_t rows, uint32_t cols) {
    const uint32_t pcols = round_up_to_tile(cols);
    std::vector<T> out(static_cast<size_t>(rows) * cols);
    for (uint32_t i = 0; i < rows; i++) {
        for (uint32_t j = 0; j < cols; j++) {
            out[i * cols + j] = a[i * pcols + j];
        }
    }
    return out;
}

// row-major -> tiles (32x32, 4 faces). rows e cols devem ser múltiplos de 32.
template <typename T>
std::vector<T> tilize(const std::vector<T> &a, uint32_t rows, uint32_t cols) {
    return convert_layout<T>(a, {rows, cols}, TensorLayoutType::LIN_ROW_MAJOR,
                             TensorLayoutType::TILED_NFACES);
}

// tiles (32x32, 4 faces) -> row-major.
template <typename T>
std::vector<T> untilize(const std::vector<T> &a, uint32_t rows, uint32_t cols) {
    return convert_layout<T>(a, {rows, cols}, TensorLayoutType::TILED_NFACES,
                             TensorLayoutType::LIN_ROW_MAJOR);
}

} // namespace sscad
