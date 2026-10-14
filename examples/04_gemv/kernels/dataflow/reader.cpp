// SPDX-License-Identifier: Apache-2.0
//
// Reader do exemplo 04 (GEMV). Lê uma vez os tiles de alpha e beta e, para
// cada linha de tiles mt atribuída ao núcleo, o tile de y e os Nt pares de
// tiles (A, x) que o compute vai multiplicar.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t y_addr = get_arg_val<uint32_t>(2);
    const uint32_t alpha_addr = get_arg_val<uint32_t>(3);
    const uint32_t beta_addr = get_arg_val<uint32_t>(4);
    const uint32_t Mt = get_arg_val<uint32_t>(5);
    const uint32_t Nt = get_arg_val<uint32_t>(6);
    const uint32_t start_row = get_arg_val<uint32_t>(7);
    const uint32_t num_rows = get_arg_val<uint32_t>(8);

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_x = tt::CBIndex::c_1;
    constexpr uint32_t cb_y = tt::CBIndex::c_2;
    constexpr uint32_t cb_alpha = tt::CBIndex::c_3;
    constexpr uint32_t cb_beta = tt::CBIndex::c_4;

    // ANCHOR: reader_accessor
    constexpr TensorAccessorArgs<0> a_args;
    const TensorAccessor a(a_args, a_addr);
    constexpr TensorAccessorArgs<a_args.next_compile_time_args_offset()> x_args;
    const TensorAccessor x(x_args, x_addr);
    constexpr TensorAccessorArgs<x_args.next_compile_time_args_offset()> y_args;
    const TensorAccessor y(y_args, y_addr);
    constexpr TensorAccessorArgs<y_args.next_compile_time_args_offset()>
        alpha_args;
    const TensorAccessor alpha(alpha_args, alpha_addr);
    constexpr TensorAccessorArgs<alpha_args.next_compile_time_args_offset()>
        beta_args;
    const TensorAccessor beta(beta_args, beta_addr);
    // ANCHOR_END: reader_accessor

    Noc noc;
    CircularBuffer a_cb(cb_a);
    CircularBuffer x_cb(cb_x);
    CircularBuffer y_cb(cb_y);
    CircularBuffer alpha_cb(cb_alpha);
    CircularBuffer beta_cb(cb_beta);

    // alpha e beta valem para todas as linhas: lidos uma vez. O compute nunca
    // os libera (cb_pop_front), então ficam disponíveis até o fim.
    alpha_cb.reserve_back(1);
    beta_cb.reserve_back(1);
    noc.async_read(alpha, alpha_cb, alpha_cb.get_tile_size(), {.page_id = 0},
                   {.offset_bytes = 0});
    noc.async_read(beta, beta_cb, beta_cb.get_tile_size(), {.page_id = 0},
                   {.offset_bytes = 0});
    noc.async_read_barrier();
    alpha_cb.push_back(1);
    beta_cb.push_back(1);

    const uint32_t a_tile_bytes = a_cb.get_tile_size();
    const uint32_t x_tile_bytes = x_cb.get_tile_size();
    const uint32_t y_tile_bytes = y_cb.get_tile_size();

    // ANCHOR: reader_loop
    const uint32_t end_row = start_row + num_rows;
    for (uint32_t mt = start_row; mt < end_row; mt++) {
        // y está na primeira coluna de tiles da matriz m x m: tile (mt, 0).
        y_cb.reserve_back(1);
        noc.async_read(y, y_cb, y_tile_bytes, {.page_id = mt * Mt},
                       {.offset_bytes = 0});
        noc.async_read_barrier();
        y_cb.push_back(1);

        for (uint32_t nt = 0; nt < Nt; nt++) {
            // Tile (mt, nt) de A (m x n) e tile (nt, mt) de x (n x m).
            const uint32_t a_tile = mt * Nt + nt;
            const uint32_t x_tile = nt * Mt + mt;

            a_cb.reserve_back(1);
            x_cb.reserve_back(1);
            noc.async_read(a, a_cb, a_tile_bytes, {.page_id = a_tile},
                           {.offset_bytes = 0});
            noc.async_read(x, x_cb, x_tile_bytes, {.page_id = x_tile},
                           {.offset_bytes = 0});
            noc.async_read_barrier();
            a_cb.push_back(1);
            x_cb.push_back(1);
        }
    }
    // ANCHOR_END: reader_loop
}
