// Reader do exemplo 06 (GEMM v2). Mesma ordem de leitura do exemplo 05, mas,
// para cada tile de saída (i, j), reserva de uma vez os Kt tiles A(i, kt) e os
// Kt tiles B(kt, j), emite as 2 Kt leituras e espera uma única barreira. As
// leituras ficam em voo ao mesmo tempo, e a latência da DRAM é paga uma vez
// por tile de saída, não uma vez por tile lido. Como os CBs têm 2 Kt tiles, o
// reader já busca os tiles do próximo tile de saída enquanto o compute ainda
// consome os do atual.

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

    constexpr TensorAccessorArgs<0> a_args;
    const TensorAccessor a(a_args, a_addr);
    constexpr TensorAccessorArgs<a_args.next_compile_time_args_offset()> b_args;
    const TensorAccessor b(b_args, b_addr);

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

        // Espaço para a linha de A e a coluna de B inteiras.
        a_cb.reserve_back(Kt);
        b_cb.reserve_back(Kt);

        // Os tiles vão para posições consecutivas do CB (offset_bytes), na
        // ordem kt = 0..Kt-1 em que o compute os consome.
        for (uint32_t kt = 0; kt < Kt; kt++) {
            noc.async_read(a, a_cb, a_tile_bytes, {.page_id = out_row * Kt + kt},
                           {.offset_bytes = kt * a_tile_bytes});
            noc.async_read(b, b_cb, b_tile_bytes, {.page_id = kt * Nt + out_col},
                           {.offset_bytes = kt * b_tile_bytes});
        }
        // Uma barreira para as 2 Kt leituras.
        noc.async_read_barrier();

        a_cb.push_back(Kt);
        b_cb.push_back(Kt);
    }
    // ANCHOR_END: reader_loop
}
