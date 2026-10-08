// SPDX-License-Identifier: Apache-2.0
//
// Kernel de movimentação de dados do exemplo 00: copia num_tiles tiles de um
// buffer da DRAM para outro, passando por um buffer de um tile na L1.
//
// As transferências pela NoC são assíncronas: noc_async_read_page e
// noc_async_write_page apenas *disparam* a transferência e retornam
// imediatamente. Quem garante que ela terminou é a barreira correspondente.
// Sem a barreira de leitura, a escrita poderia enviar um tile que ainda não
// chegou à L1; sem a barreira de escrita, a próxima leitura poderia
// sobrescrever o buffer da L1 antes de ele ter sido enviado.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t l1_addr = get_arg_val<uint32_t>(0);
    const uint32_t src_addr = get_arg_val<uint32_t>(1);
    const uint32_t dst_addr = get_arg_val<uint32_t>(2);
    const uint32_t num_tiles = get_arg_val<uint32_t>(3);

    // ANCHOR: loopback_accessor
    // Os TensorAccessorArgs vêm dos argumentos de compilação, na mesma ordem
    // em que o host os adicionou. O TensorAccessor traduz "página i" para o
    // endereço NoC (banco de DRAM + deslocamento) dessa página.
    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto src = TensorAccessor(src_args, src_addr);
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto dst = TensorAccessor(dst_args, dst_addr);
    // ANCHOR_END: loopback_accessor

    // ANCHOR: loopback_loop
    // Laço deliberadamente serial: um único tile na L1 e uma barreira após
    // cada transferência, sem sobreposição entre leitura e escrita. O exemplo
    // 02 resolve isso com circular buffers e kernels de leitura e escrita
    // separados.
    for (uint32_t i = 0; i < num_tiles; i++) {
        noc_async_read_page(i, src, l1_addr); // DRAM -> L1
        noc_async_read_barrier();

        noc_async_write_page(i, dst, l1_addr); // L1 -> DRAM
        noc_async_write_barrier();
    }
    // ANCHOR_END: loopback_loop
}
