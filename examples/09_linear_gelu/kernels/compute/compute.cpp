// Compute do exemplo 09 (camada linear + GELU), em duas fases.
//
// Fase 1 (FPU, matmul): o GEMM do exemplo 07. A cada passo kb em K, cada
// sub-bloco recarrega a soma parcial de c_24 (a partir do passo 1), soma o
// produto dos blocos de X e W e volta para c_24. Diferente do exemplo 07, o
// último passo também empacota em c_24: ao fim da fase, c_24 tem X W.
//
// Fase 2 (FPU + SFPU, fundida): para cada sub-bloco, no mesmo
// tile_regs_acquire,
//   Dst = c_24 + bias  (add_tiles_bcast_rows: a linha 0 do tile de bias é
//                       repetida nas 32 linhas do tile);
//   Dst = GELU(Dst)    (gelu_tile, na SFPU, em cima do Dst);
// e só então o resultado é empacotado em c_16 para o writer. X W + b nunca
// sai da L1, e GELU(X W + b) é escrito uma vez só.
//
// Os formatos mudam entre as fases (X, W e bias são bfloat16; c_24 é float),
// então cada mudança de operandos começa com uma reconfiguração do unpacker,
// como no exemplo 04.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/gelu.h"
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

    constexpr uint32_t x_block_tiles = per_core_M * in0_block_w;
    constexpr uint32_t w_block_tiles = in0_block_w * per_core_N;
    constexpr uint32_t subblock_tiles = out_subblock_h * out_subblock_w;
    constexpr uint32_t num_subblocks_h = per_core_M / out_subblock_h;
    constexpr uint32_t num_subblocks_w = per_core_N / out_subblock_w;

    constexpr uint32_t cb_x = tt::CBIndex::c_0;
    constexpr uint32_t cb_w = tt::CBIndex::c_1;
    constexpr uint32_t cb_bias = tt::CBIndex::c_2;
    constexpr uint32_t cb_y = tt::CBIndex::c_16;
    constexpr uint32_t cb_partial = tt::CBIndex::c_24;

    // SrcOrder::Reverse: no matmul, in0 (X) vai para SrcB e in1 (W) para SrcA.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_y);
    matmul_init(cb_x, cb_w);

    // ANCHOR: compute_matmul
    // Fase 1: c_24 = X W, sub-bloco por sub-bloco.
    for (uint32_t kb = 0; kb < num_k_blocks; kb++) {
        cb_wait_front(cb_x, x_block_tiles);
        cb_wait_front(cb_w, w_block_tiles);

        for (uint32_t sh = 0; sh < num_subblocks_h; sh++) {
            for (uint32_t sw = 0; sw < num_subblocks_w; sw++) {
                tile_regs_acquire();

                if (kb > 0) {
                    copy_tile_to_dst_init_short_with_dt(cb_w, cb_partial);
                    cb_wait_front(cb_partial, subblock_tiles);
                    for (uint32_t i = 0; i < subblock_tiles; i++) {
                        copy_tile(cb_partial, i, i);
                    }
                    cb_pop_front(cb_partial, subblock_tiles);
                    reconfig_data_format_srca(cb_partial, cb_w);
                    matmul_init(cb_x, cb_w);
                }

                uint32_t dst = 0;
                for (uint32_t h = 0; h < out_subblock_h; h++) {
                    const uint32_t row = sh * out_subblock_h + h;
                    for (uint32_t w = 0; w < out_subblock_w; w++) {
                        const uint32_t col = sw * out_subblock_w + w;
                        for (uint32_t i = 0; i < in0_block_w; i++) {
                            matmul_tiles(cb_x, cb_w, row * in0_block_w + i,
                                         i * per_core_N + col, dst);
                        }
                        dst++;
                    }
                }
                tile_regs_commit();

                cb_reserve_back(cb_partial, subblock_tiles);
                tile_regs_wait();
                for (uint32_t i = 0; i < subblock_tiles; i++) {
                    pack_tile(i, cb_partial);
                }
                tile_regs_release();
                cb_push_back(cb_partial, subblock_tiles);
            }
        }

        cb_pop_front(cb_x, x_block_tiles);
        cb_pop_front(cb_w, w_block_tiles);
    }
    // ANCHOR_END: compute_matmul

    // ANCHOR: compute_bias_gelu
    // Fase 2: Y = GELU(c_24 + bias). A soma lê c_24 (float) em SrcA e o bias
    // (bfloat16) em SrcB, na ordem normal dos operandos.
    reconfig_data_format(cb_partial, cb_bias);
    cb_wait_front(cb_bias, per_core_N);

    for (uint32_t sh = 0; sh < num_subblocks_h; sh++) {
        for (uint32_t sw = 0; sw < num_subblocks_w; sw++) {
            cb_wait_front(cb_partial, subblock_tiles);
            tile_regs_acquire();

            // O tile (h, w) do sub-bloco está na coluna sw * out_subblock_w + w
            // do bloco, e usa o tile de bias dessa coluna.
            add_bcast_rows_init(cb_partial, cb_bias);
            for (uint32_t h = 0; h < out_subblock_h; h++) {
                for (uint32_t w = 0; w < out_subblock_w; w++) {
                    const uint32_t i = h * out_subblock_w + w;
                    add_tiles_bcast_rows(cb_partial, cb_bias, i,
                                         sw * out_subblock_w + w, i);
                }
            }

            // GELU exata (fast_and_approx = false), a mesma da referência.
            gelu_tile_init<false>();
            for (uint32_t i = 0; i < subblock_tiles; i++) {
                gelu_tile<false>(i);
            }
            tile_regs_commit();
            cb_pop_front(cb_partial, subblock_tiles);

            cb_reserve_back(cb_y, subblock_tiles);
            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_tiles; i++) {
                pack_tile(i, cb_y);
            }
            tile_regs_release();
            cb_push_back(cb_y, subblock_tiles);
        }
    }

    cb_pop_front(cb_bias, per_core_N);
    // ANCHOR_END: compute_bias_gelu
}
