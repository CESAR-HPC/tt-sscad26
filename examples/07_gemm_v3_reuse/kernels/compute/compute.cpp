// Compute do exemplo 07 (GEMM v3). A cada passo kb em K recebe um bloco de A
// (per_core_M x in0_block_w) e um de B (in0_block_w x per_core_N) e percorre
// o bloco de C em sub-blocos de out_subblock_h x out_subblock_w tiles, que
// cabem no Dst:
//   - passo 0: Dst = A_blk · B_blk do sub-bloco;
//   - passos seguintes: recarrega no Dst a soma parcial guardada em c_24 e
//     soma o novo produto;
//   - último passo: empacota o sub-bloco em c_16 (para o writer); nos demais,
//     em c_24 (soma parcial).
// Os blocos de A e B só são liberados depois de usados por todos os
// sub-blocos: é esse o reuso em L1.
//
// c_24 é float, enquanto A e B são bfloat16. Para copiar a soma parcial o
// unpacker de SrcA precisa ser reconfigurado (copy_tile_to_dst_init_short_with_dt)
// e depois voltar ao formato de B (reconfig_data_format_srca), como no
// exemplo 04.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(1);
    constexpr uint32_t per_core_M = get_compile_time_arg_val(2);
    constexpr uint32_t per_core_N = get_compile_time_arg_val(3);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(4);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(5);

    constexpr uint32_t a_block_tiles = per_core_M * in0_block_w;
    constexpr uint32_t b_block_tiles = in0_block_w * per_core_N;
    constexpr uint32_t subblock_tiles = out_subblock_h * out_subblock_w;
    constexpr uint32_t num_subblocks_h = per_core_M / out_subblock_h;
    constexpr uint32_t num_subblocks_w = per_core_N / out_subblock_w;

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_c = tt::CBIndex::c_16;
    constexpr uint32_t cb_partial = tt::CBIndex::c_24;

    // SrcOrder::Reverse: no matmul, in0 (A) vai para SrcB e in1 (B) para SrcA.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_b, cb_c);
    matmul_init(cb_a, cb_b);

    // ANCHOR: compute_loop
    for (uint32_t kb = 0; kb < num_k_blocks; kb++) {
        const bool last_k_block = kb == num_k_blocks - 1;

        cb_wait_front(cb_a, a_block_tiles);
        cb_wait_front(cb_b, b_block_tiles);

        for (uint32_t sh = 0; sh < num_subblocks_h; sh++) {
            for (uint32_t sw = 0; sw < num_subblocks_w; sw++) {
                tile_regs_acquire();

                // ANCHOR: compute_reload
                if (kb > 0) {
                    // Soma parcial do sub-bloco -> Dst[0..subblock_tiles).
                    copy_tile_to_dst_init_short_with_dt(cb_b, cb_partial);
                    cb_wait_front(cb_partial, subblock_tiles);
                    for (uint32_t i = 0; i < subblock_tiles; i++) {
                        copy_tile(cb_partial, i, i);
                    }
                    cb_pop_front(cb_partial, subblock_tiles);
                    reconfig_data_format_srca(cb_partial, cb_b);
                    matmul_init(cb_a, cb_b);
                }
                // ANCHOR_END: compute_reload

                // Dst[h * out_subblock_w + w] += soma_i A(row, i) · B(i, col),
                // com índices relativos aos blocos em L1 (row-major).
                uint32_t dst = 0;
                for (uint32_t h = 0; h < out_subblock_h; h++) {
                    const uint32_t row = sh * out_subblock_h + h;
                    for (uint32_t w = 0; w < out_subblock_w; w++) {
                        const uint32_t col = sw * out_subblock_w + w;
                        for (uint32_t i = 0; i < in0_block_w; i++) {
                            matmul_tiles(cb_a, cb_b, row * in0_block_w + i,
                                         i * per_core_N + col, dst);
                        }
                        dst++;
                    }
                }
                tile_regs_commit();

                const uint32_t cb_out = last_k_block ? cb_c : cb_partial;
                cb_reserve_back(cb_out, subblock_tiles);
                tile_regs_wait();
                for (uint32_t i = 0; i < subblock_tiles; i++) {
                    pack_tile(i, cb_out);
                }
                tile_regs_release();
                cb_push_back(cb_out, subblock_tiles);
            }
        }

        cb_pop_front(cb_a, a_block_tiles);
        cb_pop_front(cb_b, b_block_tiles);
    }
    // ANCHOR_END: compute_loop
}
