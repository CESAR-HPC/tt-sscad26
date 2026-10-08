// SPDX-License-Identifier: Apache-2.0
//
// Compute do exemplo 03: idêntico ao do exemplo 02. O compute só enxerga os
// CBs, então não precisa saber quais tiles globais está processando — basta
// saber quantos.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_c = tt::CBIndex::c_16;
    constexpr uint32_t dst = 0;

    compute_kernel_hw_startup(cb_a, cb_b, cb_c);
    add_init(cb_a, cb_b);

    // ANCHOR: compute_loop
    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);

        tile_regs_acquire();
        add_tiles(cb_a, cb_b, 0, 0, dst);
        tile_regs_commit();

        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);

        cb_reserve_back(cb_c, 1);
        tile_regs_wait();
        pack_tile(dst, cb_c);
        tile_regs_release();
        cb_push_back(cb_c, 1);
    }
    // ANCHOR_END: compute_loop
}
