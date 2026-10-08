// SPDX-License-Identifier: Apache-2.0
//
// Comparação entre o resultado do dispositivo e a referência em CPU.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <fmt/core.h>

namespace sscad {

// Tolerâncias típicas. O bfloat16 tem 8 bits de mantissa (7 armazenados),
// logo o erro relativo de arredondamento é da ordem de 2^-8 ~ 0.4%.
constexpr float BF16_RTOL = 1e-2f;
constexpr float BF16_PCC = 0.999f;

struct CompareResult {
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f; // relativo à norma infinito da referência
    float pcc = 1.0f;         // coeficiente de correlação de Pearson
    bool pass = false;
};

// Coeficiente de correlação de Pearson entre dois vetores.
inline float pearson(const std::vector<float> &x, const std::vector<float> &y) {
    const size_t n = x.size();
    double mx = 0.0, my = 0.0;
    for (size_t i = 0; i < n; i++) {
        mx += x[i];
        my += y[i];
    }
    mx /= n;
    my /= n;
    double sxy = 0.0, sxx = 0.0, syy = 0.0;
    for (size_t i = 0; i < n; i++) {
        sxy += (x[i] - mx) * (y[i] - my);
        sxx += (x[i] - mx) * (x[i] - mx);
        syy += (y[i] - my) * (y[i] - my);
    }
    if (sxx == 0.0 || syy == 0.0) {
        return (sxx == syy) ? 1.0f : 0.0f;
    }
    return static_cast<float>(sxy / std::sqrt(sxx * syy));
}

// Compara `got` (dispositivo) com `expected` (CPU). Passa se o erro máximo,
// relativo ao maior valor absoluto da referência, for <= rtol e o PCC >= min_pcc.
template <typename T>
CompareResult compare(const std::vector<T> &got, const std::vector<float> &expected,
                      float rtol = BF16_RTOL, float min_pcc = BF16_PCC) {
    CompareResult r;
    if (got.size() != expected.size()) {
        fmt::print("Tamanhos diferentes: {} (dispositivo) vs {} (referência)\n", got.size(),
                   expected.size());
        return r;
    }
    std::vector<float> g(got.size());
    float ref_max = 0.0f;
    for (size_t i = 0; i < got.size(); i++) {
        g[i] = static_cast<float>(got[i]);
        r.max_abs_err = std::max(r.max_abs_err, std::abs(g[i] - expected[i]));
        ref_max = std::max(ref_max, std::abs(expected[i]));
    }
    r.max_rel_err = ref_max > 0.0f ? r.max_abs_err / ref_max : r.max_abs_err;
    r.pcc = pearson(g, expected);
    r.pass = r.max_rel_err <= rtol && r.pcc >= min_pcc;
    return r;
}

inline void print_compare(const CompareResult &r) {
    fmt::print("Erro absoluto máximo: {:.3e} | erro relativo máximo: {:.3e} | PCC: {:.6f}\n",
               r.max_abs_err, r.max_rel_err, r.pcc);
}

// Imprime PASSOU/FALHOU e devolve o código de saída do programa.
inline int report(bool pass) {
    fmt::print("{}\n", pass ? "PASSOU" : "FALHOU");
    return pass ? 0 : 1;
}

} // namespace sscad
