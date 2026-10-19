// Compute do exemplo 02: C = A + B, um tile por vez.
//
// O kernel de computação é compilado três vezes, uma para cada RISC-V que
// comanda o coprocessador Tensix: UNPACK (CB -> registradores de origem),
// MATH (FPU: soma, resultado no registrador Dst) e PACK (Dst -> CB). As
// chamadas abaixo são as mesmas nos três; cada um executa a sua parte.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    const uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_c = tt::CBIndex::c_16;
    constexpr uint32_t dst = 0; // índice do tile no registrador Dst

    // ANCHOR: compute_init
    // Configura unpacker, FPU e packer para os formatos dos CBs usados e,
    // em seguida, a FPU para a operação de soma.
    compute_kernel_hw_startup(cb_a, cb_b, cb_c);
    add_init(cb_a, cb_b);
    // ANCHOR_END: compute_init

    // ANCHOR: compute_loop
    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);

        // Handshake do registrador Dst entre MATH e PACK:
        //  - tile_regs_acquire: MATH obtém o Dst para escrever;
        //  - tile_regs_commit:  MATH entrega o Dst preenchido ao PACK;
        //  - tile_regs_wait:    PACK espera o Dst ficar pronto;
        //  - tile_regs_release: PACK devolve o Dst para o próximo MATH.
        tile_regs_acquire();
        add_tiles(cb_a, cb_b, 0, 0, dst);
        tile_regs_commit();

        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);

        cb_reserve_back(cb_c, 1);
        tile_regs_wait();
        pack_tile(dst, cb_c);
        tile_regs_release();
        cb_push_back(cb_c, 1);
    }
    // ANCHOR_END: compute_loop
}
