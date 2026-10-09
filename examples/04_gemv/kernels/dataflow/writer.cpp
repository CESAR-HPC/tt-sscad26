// SPDX-License-Identifier: Apache-2.0
//
// Writer do exemplo 04 (GEMV). Escreve um tile de y por linha de tiles,
// na mesma posição que ele ocupa na matriz de y: o tile (mt, 0).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t Mt = get_arg_val<uint32_t>(1);
    const uint32_t start_row = get_arg_val<uint32_t>(2);
    const uint32_t num_rows = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr TensorAccessorArgs<0> out_args;
    const TensorAccessor out(out_args, out_addr);

    // ANCHOR: writer_loop
    const uint32_t end_row = start_row + num_rows;
    for (uint32_t mt = start_row; mt < end_row; mt++) {
        cb_wait_front(cb_out, 1);
        noc_async_write_page(mt * Mt, out, get_read_ptr(cb_out));
        noc_async_writes_flushed();
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
    // ANCHOR_END: writer_loop
}
