// SPDX-License-Identifier: Apache-2.0
//
// Reader do exemplo 05 (GEMM v1). Para cada tile de saída (i, j) da faixa do
// núcleo, lê os Kt tiles A(i, kt) e B(kt, j), na ordem em que o compute os
// consome. Cada leitura tem a sua barreira: o reader fica parado esperando a
// DRAM a cada tile. Os mesmos tiles de A e B são lidos de novo para cada
// tile de saída que precisa deles (nenhum reuso).

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t Kt = get_arg_val<uint32_t>(2);
    const uint32_t Nt = get_arg_val<uint32_t>(3);
    const uint32_t start_tile = get_arg_val<uint32_t>(4);
    const uint32_t num_tiles = get_arg_val<uint32_t>(5);

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
    const uint32_t end_tile = start_tile + num_tiles;
    for (uint32_t t = start_tile; t < end_tile; t++) {
        const uint32_t out_row = t / Nt;
        const uint32_t out_col = t % Nt;

        for (uint32_t kt = 0; kt < Kt; kt++) {
            // A é Mt x Kt tiles: o tile (out_row, kt) é out_row * Kt + kt.
            a_cb.reserve_back(1);
            noc.async_read(a, a_cb, a_tile_bytes, {.page_id = out_row * Kt + kt},
                           {.offset_bytes = 0});
            noc.async_read_barrier();
            a_cb.push_back(1);

            // B é Kt x Nt tiles: o tile (kt, out_col) é kt * Nt + out_col.
            b_cb.reserve_back(1);
            noc.async_read(b, b_cb, b_tile_bytes, {.page_id = kt * Nt + out_col},
                           {.offset_bytes = 0});
            noc.async_read_barrier();
            b_cb.push_back(1);
        }
    }
    // ANCHOR_END: reader_loop
}
