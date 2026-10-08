// SPDX-License-Identifier: Apache-2.0
//
// Writer do exemplo 03: igual ao do exemplo 02, mas escreve apenas a faixa
// [start_tile, start_tile + num_tiles) atribuída a este núcleo.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t c_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_c = tt::CBIndex::c_16;

    constexpr TensorAccessorArgs<0> c_args;
    const TensorAccessor c(c_args, c_addr);

    // ANCHOR: writer_loop
    const uint32_t end_tile = start_tile + num_tiles;
    for (uint32_t i = start_tile; i < end_tile; i++) {
        cb_wait_front(cb_c, 1);
        noc_async_write_page(i, c, get_read_ptr(cb_c));
        noc_async_writes_flushed();
        cb_pop_front(cb_c, 1);
    }
    noc_async_write_barrier();
    // ANCHOR_END: writer_loop
}
