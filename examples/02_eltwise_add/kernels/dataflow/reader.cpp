// Reader do exemplo 02: traz os tiles de A e B da DRAM para os CBs c_0 e c_1.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
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
    constexpr TensorAccessorArgs<0> a_args;
    const TensorAccessor a(a_args, a_addr);
    constexpr TensorAccessorArgs<a_args.next_compile_time_args_offset()> b_args;
    const TensorAccessor b(b_args, b_addr);
    // ANCHOR_END: reader_accessor

    Noc noc;
    CircularBuffer a_cb(cb_a);
    CircularBuffer b_cb(cb_b);
    const uint32_t a_tile_bytes = a_cb.get_tile_size();
    const uint32_t b_tile_bytes = b_cb.get_tile_size();

    // ANCHOR: reader_loop
    for (uint32_t i = 0; i < num_tiles; i++) {
        // Bloqueia até haver um espaço livre em cada CB (o compute libera
        // espaço ao consumir os tiles).
        a_cb.reserve_back(1);
        b_cb.reserve_back(1);

        // As duas leituras são disparadas antes da barreira e acontecem em
        // paralelo na NoC.
        noc.async_read(a, a_cb, a_tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read(b, b_cb, b_tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();

        // ANCHOR: reader_dprint
#ifdef SSCAD_DPRINT
        DPRINT("reader: tile {} de {} lido\n", i, num_tiles);
#endif
        // ANCHOR_END: reader_dprint

        // Publica os tiles: a partir daqui o compute os vê em cb_wait_front.
        a_cb.push_back(1);
        b_cb.push_back(1);
    }
    // ANCHOR_END: reader_loop
}
