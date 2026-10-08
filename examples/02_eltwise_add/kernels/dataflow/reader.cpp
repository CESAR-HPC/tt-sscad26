// SPDX-License-Identifier: Apache-2.0
//
// Reader do exemplo 02: traz os tiles de A e B da DRAM para os CBs c_0 e c_1.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#ifdef SSCAD_DPRINT
#include "api/debug/dprint.h"
#endif

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;

    // ANCHOR: reader_accessor
    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, a_addr);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr);
    // ANCHOR_END: reader_accessor

    // ANCHOR: reader_loop
    for (uint32_t i = 0; i < num_tiles; i++) {
        // Bloqueia até haver um espaço livre em cada CB (o compute libera
        // espaço ao consumir os tiles).
        cb_reserve_back(cb_a, 1);
        cb_reserve_back(cb_b, 1);

        // As duas leituras são disparadas antes da barreira e acontecem em
        // paralelo na NoC.
        noc_async_read_page(i, a, get_write_ptr(cb_a));
        noc_async_read_page(i, b, get_write_ptr(cb_b));
        noc_async_read_barrier();

#ifdef SSCAD_DPRINT
        DPRINT("reader: tile {} de {} lido\n", i, num_tiles);
#endif

        // Publica os tiles: a partir daqui o compute os vê em cb_wait_front.
        cb_push_back(cb_a, 1);
        cb_push_back(cb_b, 1);
    }
    // ANCHOR_END: reader_loop
}
