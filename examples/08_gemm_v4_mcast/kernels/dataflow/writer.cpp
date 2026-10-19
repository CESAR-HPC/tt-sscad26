// SPDX-License-Identifier: Apache-2.0
//
// Writer do exemplo 08 (GEMM v4), idêntico ao do exemplo 07. O compute
// entrega o bloco (by, bx) de C sub-bloco por sub-bloco (out_subblock_h x
// out_subblock_w tiles, row-major dentro do sub-bloco, sub-blocos em row-major
// dentro do bloco). O writer calcula a posição de cada tile em C e o escreve
// na DRAM.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t c_addr = get_arg_val<uint32_t>(0);
    const uint32_t Nt = get_arg_val<uint32_t>(1);
    const uint32_t block_y = get_arg_val<uint32_t>(2);
    const uint32_t block_x = get_arg_val<uint32_t>(3);
    const uint32_t per_core_M = get_arg_val<uint32_t>(4);
    const uint32_t per_core_N = get_arg_val<uint32_t>(5);
    const uint32_t out_subblock_h = get_arg_val<uint32_t>(6);
    const uint32_t out_subblock_w = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_c = tt::CBIndex::c_16;

    constexpr TensorAccessorArgs<0> c_args;
    const TensorAccessor c(c_args, c_addr);

    Noc noc;
    CircularBuffer c_cb(cb_c);
    const uint32_t tile_bytes = c_cb.get_tile_size();

    const uint32_t subblock_tiles = out_subblock_h * out_subblock_w;
    const uint32_t num_subblocks_h = per_core_M / out_subblock_h;
    const uint32_t num_subblocks_w = per_core_N / out_subblock_w;

    // ANCHOR: writer_loop
    for (uint32_t sh = 0; sh < num_subblocks_h; sh++) {
        for (uint32_t sw = 0; sw < num_subblocks_w; sw++) {
            c_cb.wait_front(subblock_tiles);
            uint32_t slot = 0;
            for (uint32_t h = 0; h < out_subblock_h; h++) {
                const uint32_t row =
                    block_y * per_core_M + sh * out_subblock_h + h;
                for (uint32_t w = 0; w < out_subblock_w; w++) {
                    const uint32_t col =
                        block_x * per_core_N + sw * out_subblock_w + w;
                    noc.async_write(c_cb, c, tile_bytes,
                                    {.offset_bytes = slot * tile_bytes},
                                    {.page_id = row * Nt + col});
                    slot++;
                }
            }
            noc.async_writes_flushed();
            c_cb.pop_front(subblock_tiles);
        }
    }
    noc.async_write_barrier();
    // ANCHOR_END: writer_loop
}
