// SPDX-License-Identifier: Apache-2.0
//
// Writer do exemplo 04 (GEMV). Escreve um tile de y por linha de tiles,
// na mesma posição que ele ocupa na matriz de y: o tile (mt, 0).

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t Mt = get_arg_val<uint32_t>(1);
    const uint32_t start_row = get_arg_val<uint32_t>(2);
    const uint32_t num_rows = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr TensorAccessorArgs<0> out_args;
    const TensorAccessor out(out_args, out_addr);

    Noc noc;
    CircularBuffer out_cb(cb_out);
    const uint32_t tile_bytes = out_cb.get_tile_size();

    // ANCHOR: writer_loop
    const uint32_t end_row = start_row + num_rows;
    for (uint32_t mt = start_row; mt < end_row; mt++) {
        out_cb.wait_front(1);
        noc.async_write(out_cb, out, tile_bytes, {.offset_bytes = 0},
                        {.page_id = mt * Mt});
        noc.async_writes_flushed();
        out_cb.pop_front(1);
    }
    noc.async_write_barrier();
    // ANCHOR_END: writer_loop
}
