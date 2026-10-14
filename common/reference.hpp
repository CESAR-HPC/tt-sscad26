// SPDX-License-Identifier: Apache-2.0
//
// Implementações de referência em CPU. Todas acumulam em float, de modo que o
// resultado do dispositivo possa ser comparado com uma versão "exata".
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace sscad::reference {

// C = A + B, elemento a elemento.
template <typename T>
std::vector<float> eltwise_add(const std::vector<T> &a, const std::vector<T> &b) {
    std::vector<float> c(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        c[i] = static_cast<float>(a[i]) + static_cast<float>(b[i]);
    }
    return c;
}

// y = alpha * A x + beta * y, com A m x n em row-major.
template <typename T>
std::vector<float> gemv(uint32_t m, uint32_t n, float alpha, const std::vector<T> &a,
                        const std::vector<T> &x, float beta, const std::vector<float> &y) {
    std::vector<float> out(m);
    for (uint32_t i = 0; i < m; i++) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < n; j++) {
            acc += static_cast<float>(a[i * n + j]) * static_cast<float>(x[j]);
        }
        out[i] = alpha * acc + beta * y[i];
    }
    return out;
}

// C = A B, com A m x k e B k x n em row-major.
template <typename T>
std::vector<float> gemm(uint32_t m, uint32_t n, uint32_t k, const std::vector<T> &a,
                        const std::vector<T> &b) {
    std::vector<float> c(static_cast<size_t>(m) * n, 0.0f);
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t p = 0; p < k; p++) {
            const float aip = static_cast<float>(a[i * k + p]);
            for (uint32_t j = 0; j < n; j++) {
                c[i * n + j] += aip * static_cast<float>(b[p * n + j]);
            }
        }
    }
    return c;
}

// GELU exata: x * Phi(x).
inline float gelu(float x) { return 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f))); }

// Y = GELU(X W + b), com X m x k, W k x n e b de tamanho n.
template <typename T>
std::vector<float> linear_gelu(uint32_t m, uint32_t n, uint32_t k, const std::vector<T> &x,
                               const std::vector<T> &w, const std::vector<T> &bias) {
    std::vector<float> y = gemm(m, n, k, x, w);
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < n; j++) {
            y[i * n + j] = gelu(y[i * n + j] + static_cast<float>(bias[j]));
        }
    }
    return y;
}

} // namespace sscad::reference
