// Reader do exemplo 07 (GEMM v3). O núcleo calcula o bloco (by, bx) de C, de
// per_core_M x per_core_N tiles. Para cada passo kb em K, lê para a L1:
//   - o bloco de A com as linhas de tiles by * per_core_M + h e as colunas
//     kb * in0_block_w + w (per_core_M x in0_block_w tiles);
//   - o bloco de B com as linhas kb * in0_block_w + h e as colunas
//     bx * per_core_N + w (in0_block_w x per_core_N tiles);
// ambos em row-major dentro do CB, com uma barreira por par de blocos. Cada
// tile é lido da DRAM uma única vez por núcleo.

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
    const uint32_t block_y = get_arg_val<uint32_t>(4);
    const uint32_t block_x = get_arg_val<uint32_t>(5);
    const uint32_t per_core_M = get_arg_val<uint32_t>(6);
    const uint32_t per_core_N = get_arg_val<uint32_t>(7);
    const uint32_t in0_block_w = get_arg_val<uint32_t>(8);

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

    const uint32_t a_block_tiles = per_core_M * in0_block_w;
    const uint32_t b_block_tiles = in0_block_w * per_core_N;
    const uint32_t num_k_blocks = Kt / in0_block_w;
    // Primeira linha de tiles de A e primeira coluna de tiles de B do bloco.
    const uint32_t a_row0 = block_y * per_core_M;
    const uint32_t b_col0 = block_x * per_core_N;

    // ANCHOR: reader_loop
    for (uint32_t kb = 0; kb < num_k_blocks; kb++) {
        const uint32_t k0 = kb * in0_block_w;

        a_cb.reserve_back(a_block_tiles);
        b_cb.reserve_back(b_block_tiles);

        uint32_t slot = 0;
        for (uint32_t h = 0; h < per_core_M; h++) {
            for (uint32_t w = 0; w < in0_block_w; w++) {
                const uint32_t tile = (a_row0 + h) * Kt + (k0 + w);
                noc.async_read(a, a_cb, a_tile_bytes, {.page_id = tile},
                               {.offset_bytes = slot * a_tile_bytes});
                slot++;
            }
        }

        slot = 0;
        for (uint32_t h = 0; h < in0_block_w; h++) {
            for (uint32_t w = 0; w < per_core_N; w++) {
                const uint32_t tile = (k0 + h) * Nt + (b_col0 + w);
                noc.async_read(b, b_cb, b_tile_bytes, {.page_id = tile},
                               {.offset_bytes = slot * b_tile_bytes});
                slot++;
            }
        }

        noc.async_read_barrier();
        a_cb.push_back(a_block_tiles);
        b_cb.push_back(b_block_tiles);
    }
    // ANCHOR_END: reader_loop
}
