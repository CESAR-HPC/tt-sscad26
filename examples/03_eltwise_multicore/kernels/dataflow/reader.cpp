// Reader do exemplo 03: igual ao do exemplo 02, mas lê apenas a faixa
// [start_tile, start_tile + num_tiles) atribuída a este núcleo.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_tile = get_arg_val<uint32_t>(2);
    const uint32_t num_tiles = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;

    constexpr TensorAccessorArgs<0> a_args;
    const TensorAccessor a(a_args, a_addr);
    constexpr TensorAccessorArgs<a_args.next_compile_time_args_offset()> b_args;
    const TensorAccessor b(b_args, b_addr);

    Noc noc;
    CircularBuffer a_cb(cb_a);
    CircularBuffer b_cb(cb_b);
    const uint32_t a_tile_bytes = a_cb.get_tile_size();
    const uint32_t b_tile_bytes = b_cb.get_tile_size();

    // ANCHOR: reader_loop
    const uint32_t end_tile = start_tile + num_tiles;
    for (uint32_t i = start_tile; i < end_tile; i++) {
        a_cb.reserve_back(1);
        b_cb.reserve_back(1);

        noc.async_read(a, a_cb, a_tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read(b, b_cb, b_tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();

        a_cb.push_back(1);
        b_cb.push_back(1);
    }
    // ANCHOR_END: reader_loop
}
