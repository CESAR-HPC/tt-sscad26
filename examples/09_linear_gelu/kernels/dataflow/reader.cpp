// Reader do exemplo 09 (camada linear + GELU). Igual ao reader do exemplo 07
// (com X no lugar de A e W no lugar de B), mais os tiles do bias: antes do
// laço em K, lê uma vez os per_core_N tiles de bias das colunas
// bx * per_core_N + w do bloco. O compute os mantém no CB até o fim.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t bias_addr = get_arg_val<uint32_t>(2);
    const uint32_t Kt = get_arg_val<uint32_t>(3);
    const uint32_t Nt = get_arg_val<uint32_t>(4);
    const uint32_t block_y = get_arg_val<uint32_t>(5);
    const uint32_t block_x = get_arg_val<uint32_t>(6);
    const uint32_t per_core_M = get_arg_val<uint32_t>(7);
    const uint32_t per_core_N = get_arg_val<uint32_t>(8);
    const uint32_t in0_block_w = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_x = tt::CBIndex::c_0;
    constexpr uint32_t cb_w = tt::CBIndex::c_1;
    constexpr uint32_t cb_bias = tt::CBIndex::c_2;

    constexpr TensorAccessorArgs<0> x_args;
    const TensorAccessor x(x_args, x_addr);
    constexpr TensorAccessorArgs<x_args.next_compile_time_args_offset()> w_args;
    const TensorAccessor w(w_args, w_addr);
    constexpr TensorAccessorArgs<w_args.next_compile_time_args_offset()>
        bias_args;
    const TensorAccessor bias(bias_args, bias_addr);

    Noc noc;
    CircularBuffer x_cb(cb_x);
    CircularBuffer w_cb(cb_w);
    CircularBuffer bias_cb(cb_bias);
    const uint32_t x_tile_bytes = x_cb.get_tile_size();
    const uint32_t w_tile_bytes = w_cb.get_tile_size();
    const uint32_t bias_tile_bytes = bias_cb.get_tile_size();

    const uint32_t x_block_tiles = per_core_M * in0_block_w;
    const uint32_t w_block_tiles = in0_block_w * per_core_N;
    const uint32_t num_k_blocks = Kt / in0_block_w;
    const uint32_t x_row0 = block_y * per_core_M;
    const uint32_t w_col0 = block_x * per_core_N;

    // ANCHOR: reader_bias
    // O bias tem uma única linha de tiles: o tile da coluna j é o tile j.
    bias_cb.reserve_back(per_core_N);
    for (uint32_t j = 0; j < per_core_N; j++) {
        noc.async_read(bias, bias_cb, bias_tile_bytes, {.page_id = w_col0 + j},
                       {.offset_bytes = j * bias_tile_bytes});
    }
    noc.async_read_barrier();
    bias_cb.push_back(per_core_N);
    // ANCHOR_END: reader_bias

    // ANCHOR: reader_loop
    for (uint32_t kb = 0; kb < num_k_blocks; kb++) {
        const uint32_t k0 = kb * in0_block_w;

        x_cb.reserve_back(x_block_tiles);
        w_cb.reserve_back(w_block_tiles);

        uint32_t slot = 0;
        for (uint32_t h = 0; h < per_core_M; h++) {
            for (uint32_t i = 0; i < in0_block_w; i++) {
                const uint32_t tile = (x_row0 + h) * Kt + (k0 + i);
                noc.async_read(x, x_cb, x_tile_bytes, {.page_id = tile},
                               {.offset_bytes = slot * x_tile_bytes});
                slot++;
            }
        }

        slot = 0;
        for (uint32_t h = 0; h < in0_block_w; h++) {
            for (uint32_t j = 0; j < per_core_N; j++) {
                const uint32_t tile = (k0 + h) * Nt + (w_col0 + j);
                noc.async_read(w, w_cb, w_tile_bytes, {.page_id = tile},
                               {.offset_bytes = slot * w_tile_bytes});
                slot++;
            }
        }

        noc.async_read_barrier();
        x_cb.push_back(x_block_tiles);
        w_cb.push_back(w_block_tiles);
    }
    // ANCHOR_END: reader_loop
}
