// Compute do exemplo 05 (GEMM v1). Para cada tile de saída: zera o Dst,
// acumula os Kt produtos A(i, kt) · B(kt, j) com matmul_tiles e empacota o
// resultado. O compute não sabe quais tiles está processando — só quantos.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"

void kernel_main() {
    const uint32_t Kt = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_c = tt::CBIndex::c_16;
    constexpr uint32_t dst = 0;

    // SrcOrder::Reverse: no matmul, in0 (A) vai para SrcB e in1 (B) para SrcA.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_b, cb_c);
    matmul_init(cb_a, cb_b);

    // ANCHOR: compute_loop
    for (uint32_t t = 0; t < num_tiles; t++) {
        // tile_regs_acquire zera o Dst; cada matmul_tiles soma um produto a ele.
        tile_regs_acquire();
        for (uint32_t kt = 0; kt < Kt; kt++) {
            cb_wait_front(cb_a, 1);
            cb_wait_front(cb_b, 1);
            matmul_tiles(cb_a, cb_b, 0, 0, dst);
            cb_pop_front(cb_a, 1);
            cb_pop_front(cb_b, 1);
        }
        tile_regs_commit();

        cb_reserve_back(cb_c, 1);
        tile_regs_wait();
        pack_tile(dst, cb_c);
        tile_regs_release();
        cb_push_back(cb_c, 1);
    }
    // ANCHOR_END: compute_loop
}
