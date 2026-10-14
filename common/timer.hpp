// SPDX-License-Identifier: Apache-2.0
//
// Medição de tempo de parede no host.
#pragma once

#include <chrono>

namespace sscad {

class Timer {
public:
    Timer() : start_(Clock::now()) {}
    void reset() { start_ = Clock::now(); }
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
    }

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point start_;
};

} // namespace sscad
