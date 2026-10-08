// SPDX-License-Identifier: Apache-2.0
//
// Kernel do exemplo 01: soma dois uint32 usando o próprio RISC-V de
// movimentação de dados como CPU.
//
// Limitações desses núcleos ("baby RISC-V", ver
// tt-isa-documentation/WormholeB0/TensixTile/BabyRISCV/README.md):
//  - cada Tensix tem 5 RISC-V em ordem, de emissão simples, RV32IM, ~1 GHz;
//    dois deles (B e NC) executam os kernels de movimentação de dados e os
//    outros três comandam os motores de computação;
//  - não há unidade de ponto flutuante: float/bfloat16 seria emulado em
//    software; só aritmética inteira escalar é barata;
//  - processam um valor por vez, ordens de grandeza abaixo dos motores
//    matricial (FPU) e vetorial (SFPU) usados pelos kernels de computação;
//  - memória local minúscula (2–4 KiB de dados); os dados ficam na L1;
//  - não acessam a DRAM diretamente (tudo passa pela NoC até a L1) e não há
//    ordenação automática entre os stores do núcleo e as transferências da NoC.
// Use-os para controle, cálculo de endereços e contabilidade; deixe a
// aritmética sobre os dados para o kernel de computação (exemplo 02).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "ckernel.h"

void kernel_main() {
    const uint32_t a_dram = get_arg_val<uint32_t>(0);
    const uint32_t b_dram = get_arg_val<uint32_t>(1);
    const uint32_t c_dram = get_arg_val<uint32_t>(2);
    const uint32_t a_l1 = get_arg_val<uint32_t>(3);
    const uint32_t b_l1 = get_arg_val<uint32_t>(4);
    const uint32_t c_l1 = get_arg_val<uint32_t>(5);

    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    const auto a = TensorAccessor(a_args, a_dram);
    const auto b = TensorAccessor(b_args, b_dram);
    const auto c = TensorAccessor(c_args, c_dram);

    // ANCHOR: riscv_read
    // As duas leituras são disparadas juntas e esperadas por uma única
    // barreira: as transferências se sobrepõem na NoC.
    noc_async_read_page(0, a, a_l1);
    noc_async_read_page(0, b, b_l1);
    noc_async_read_barrier();
    // ANCHOR_END: riscv_read

    // ANCHOR: riscv_add
    // A L1 está mapeada no espaço de endereços do RISC-V: basta um ponteiro.
    volatile tt_l1_ptr uint32_t *a_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(a_l1);
    volatile tt_l1_ptr uint32_t *b_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(b_l1);
    volatile tt_l1_ptr uint32_t *c_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(c_l1);

    DPRINT("RISC-V: {} + {} = {}\n", *a_ptr, *b_ptr, *a_ptr + *b_ptr);
    *c_ptr = *a_ptr + *b_ptr;

    // O store acima pode ser "aposentado" pelo RISC-V antes de chegar à L1, e
    // o RISC-V e a NoC são clientes independentes da L1, sem ordem entre si.
    // A instrução `fence` não ajuda: nesses núcleos ela é um nop
    // (BabyRISCV/MemoryOrdering.md). load_blocking relê o endereço e trava o
    // pipeline até o store ter sido processado, garantindo que a NoC envie o
    // valor novo.
    (void)ckernel::load_blocking(c_ptr);
    // ANCHOR_END: riscv_add

    // ANCHOR: riscv_write
    noc_async_write_page(0, c, c_l1);
    noc_async_write_barrier();
    // ANCHOR_END: riscv_write
}
