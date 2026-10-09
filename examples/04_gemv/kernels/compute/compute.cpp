// SPDX-License-Identifier: Apache-2.0
//
// Compute do exemplo 04 (GEMV). Para cada linha de tiles, em duas fases:
//   1. FPU: Dst = soma_nt A(mt, nt) · x(nt, mt) (matmul_tiles acumula no Dst);
//      o tile resultante vai para o CB intermediário c_5.
//   2. SFPU: y = alpha * (A x) + beta * y, elemento a elemento, com os quatro
//      tiles (y, beta, A x, alpha) copiados para o Dst.
//
// As duas fases leem dados de formatos diferentes (A e x em bfloat16, os
// demais em float). O unpacker não se reconfigura sozinho, então cada fase
// começa com reconfig_data_format* para os CBs que vai ler. O packer escreve
// sempre float (c_5 e c_16), e não precisa ser reconfigurado.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    const uint32_t Nt = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_x = tt::CBIndex::c_1;
    constexpr uint32_t cb_y = tt::CBIndex::c_2;
    constexpr uint32_t cb_alpha = tt::CBIndex::c_3;
    constexpr uint32_t cb_beta = tt::CBIndex::c_4;
    constexpr uint32_t cb_ax = tt::CBIndex::c_5;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    // Registradores do Dst usados na fase 2 (mesma ordem do tt-blas). Com
    // fp32_dest_acc_en cabem 4 tiles FP32 no Dst.
    constexpr uint32_t dst_y = 0;
    constexpr uint32_t dst_beta = 1;
    constexpr uint32_t dst_ax = 2;
    constexpr uint32_t dst_alpha = 3;

    // SrcOrder::Reverse: no matmul, in0 (A) vai para SrcB e in1 (x) para SrcA.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_x, cb_ax);

    cb_wait_front(cb_alpha, 1);
    cb_wait_front(cb_beta, 1);

    for (uint32_t row = 0; row < num_rows; row++) {
        // ANCHOR: compute_matmul
        // Fase 1 (FPU): A x da linha de tiles. tile_regs_acquire zera o Dst,
        // e cada matmul_tiles soma o produto de um par de tiles ao Dst.
        reconfig_data_format<SrcOrder::Reverse>(cb_a, cb_x);
        matmul_init(cb_a, cb_x);

        tile_regs_acquire();
        for (uint32_t nt = 0; nt < Nt; nt++) {
            cb_wait_front(cb_a, 1);
            cb_wait_front(cb_x, 1);
            matmul_tiles(cb_a, cb_x, 0, 0, dst_y);
            cb_pop_front(cb_a, 1);
            cb_pop_front(cb_x, 1);
        }
        tile_regs_commit();

        cb_reserve_back(cb_ax, 1);
        tile_regs_wait();
        pack_tile(dst_y, cb_ax);
        tile_regs_release();
        cb_push_back(cb_ax, 1);
        // ANCHOR_END: compute_matmul

        // ANCHOR: compute_axpby
        // Fase 2 (SFPU): copia os quatro tiles float para o Dst e combina.
        reconfig_data_format_srca(cb_y);
        copy_tile_init(cb_y);

        cb_wait_front(cb_y, 1);
        cb_wait_front(cb_ax, 1);

        tile_regs_acquire();
        copy_tile(cb_y, 0, dst_y);
        copy_tile(cb_beta, 0, dst_beta);
        copy_tile(cb_ax, 0, dst_ax);
        copy_tile(cb_alpha, 0, dst_alpha);

        mul_binary_tile_init();
        mul_binary_tile(dst_ax, dst_alpha, dst_ax); // alpha * (A x)
        mul_binary_tile(dst_y, dst_beta, dst_y);    // beta * y
        add_binary_tile_init();
        add_binary_tile(dst_y, dst_ax, dst_y);      // alpha * (A x) + beta * y
        tile_regs_commit();

        cb_pop_front(cb_y, 1);
        cb_pop_front(cb_ax, 1);

        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(dst_y, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        // ANCHOR_END: compute_axpby
    }
}
