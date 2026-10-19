// Writer do exemplo 06 (GEMM v2). Em vez de esperar cada escrita terminar
// (noc_async_write_barrier por tile, como no exemplo 05), espera só que a
// escrita tenha saído da L1 (async_writes_flushed): a partir daí o slot do
// CB pode ser liberado para o compute, mesmo que o dado ainda esteja a
// caminho da DRAM. Uma barreira no fim garante que todas as escritas
// terminaram antes de o kernel acabar.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t c_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_c = tt::CBIndex::c_16;

    constexpr TensorAccessorArgs<0> c_args;
    const TensorAccessor c(c_args, c_addr);

    Noc noc;
    CircularBuffer c_cb(cb_c);
    const uint32_t tile_bytes = c_cb.get_tile_size();

    // ANCHOR: writer_loop
    const uint32_t end_tile = start_tile + num_tiles;
    for (uint32_t t = start_tile; t < end_tile; t++) {
        c_cb.wait_front(1);
        noc.async_write(c_cb, c, tile_bytes, {.offset_bytes = 0}, {.page_id = t});
        noc.async_writes_flushed();
        c_cb.pop_front(1);
    }
    noc.async_write_barrier();
    // ANCHOR_END: writer_loop
}
