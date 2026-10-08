# 01 — Soma de inteiros no RISC-V

Os processadores que executam os kernels de movimentação de dados são CPUs
RISC-V comuns. Este exemplo mostra que eles executam C++ normal: o kernel lê
dois `uint32` da DRAM, soma-os e escreve o resultado de volta. O objetivo
principal, porém, é deixar claras as **limitações** desses núcleos.

Referência: `tt-metal/tt_metal/programming_examples/add_2_integers_in_riscv`
(cujo kernel usa o `InterleavedAddrGen`, obsoleto; aqui o acesso à DRAM é
feito com `TensorAccessor`).

## Mapeamento

| Item | Valor |
|---|---|
| Operação | `c = a + b`, `uint32`, com `a = 14` e `b = 7` |
| Núcleos | 1 (`{0,0}`) |
| Kernels | 1 de movimentação de dados no RISC-V 0: `kernels/dataflow/riscv_add.cpp` |
| Buffers | 3 na DRAM (`a`, `b`, `c`) + 3 na L1 (cópias de trabalho), página = 4 bytes |
| Validação | `a + b` calculado no host |

O kernel tem três partes:

1. **leitura**: duas `noc_async_read_page` e uma única barreira;
2. **soma**: a L1 está mapeada no espaço de endereços do RISC-V;
   `CoreLocalMem<volatile uint32_t>` dá acesso tipado a um endereço da L1,
   sem ponteiros. Depois do store, `ckernel::load_blocking`
   garante que o valor chegou à L1 antes de a NoC lê-lo: o RISC-V e a NoC são
   clientes independentes da L1 e a instrução `fence` é um nop nesses núcleos;
3. **escrita**: `noc_async_write_page` e a barreira.

## Limitações do RISC-V

Fonte: `tt-isa-documentation/WormholeB0/TensixTile/BabyRISCV/README.md`.

- Cada Tensix tem **5 RISC-V "baby"**: em ordem, emissão simples,
  **RV32IM**, ~1 GHz. Dois deles (B e NC) executam os kernels de
  movimentação de dados; os outros três (T0, T1, T2) comandam o
  coprocessador Tensix nos kernels de computação (UNPACK, MATH, PACK).
- **Sem unidade de ponto flutuante**: `float` e bfloat16 seriam emulados em
  software. Só a aritmética inteira escalar é barata.
- **Escalares**: processam um valor por vez, ordens de grandeza abaixo dos
  motores matricial (FPU) e vetorial (SFPU) usados pelos kernels de
  computação, que operam sobre tiles inteiros de 32x32.
- **Memória local minúscula**: 2–4 KiB de RAM de dados e um pequeno cache de
  instruções. Os dados ficam na L1.
- **Sem acesso direto à DRAM** (tudo passa pela NoC até a L1) e **sem
  ordenação automática** entre os stores do núcleo e as transferências da NoC.

Conclusão: use os RISC-V para controle, cálculo de endereços e contabilidade
(índices de laço, aritmética de endereços, pequenas contas inteiras) e deixe
a aritmética sobre os dados para o kernel de computação — próximo exemplo.

## Como executar

```bash
./scripts/run.sh 01_riscv_add
TT_METAL_DPRINT_CORES=0,0 ./scripts/run.sh 01_riscv_add
```

Com `TT_METAL_DPRINT_CORES=0,0`, a mensagem do `DPRINT` do kernel aparece
(funciona no simulador):

```
Exemplo 01: 14 + 7 no RISC-V de movimentação de dados
0:0-0:BR: RISC-V: 14 + 7 = 21
Resultado: 21 (esperado 21)
PASSOU
```

Próximo passo: [02_eltwise_add](../02_eltwise_add) — a soma feita pelo
kernel de computação.
