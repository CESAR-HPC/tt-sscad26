// SPDX-License-Identifier: Apache-2.0
//
// Writer do exemplo 02: leva os tiles de C do CB c_16 para a DRAM.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t c_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_c = tt::CBIndex::c_16;

    constexpr auto c_args = TensorAccessorArgs<0>();
    const auto c = TensorAccessor(c_args, c_addr);

    // ANCHOR: writer_loop
    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_c, 1);
        noc_async_write_page(i, c, get_read_ptr(cb_c));

        // writes_flushed espera apenas que o tile tenha *saído* da L1, o que
        // basta para liberar o espaço no CB. Não esperamos a escrita chegar à
        // DRAM a cada tile: isso é feito uma única vez, no fim.
        noc_async_writes_flushed();
        cb_pop_front(cb_c, 1);
    }
    noc_async_write_barrier();
    // ANCHOR_END: writer_loop
}
