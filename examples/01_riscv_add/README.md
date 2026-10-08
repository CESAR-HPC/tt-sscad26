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
| Operação | `c = a + b`, `uint32` (padrão `14 + 7`) |
| Núcleos | 1 (`{0,0}`) |
| Kernels | 1 de movimentação de dados no RISC-V 0: `kernels/dataflow/riscv_add.cpp` |
| Buffers | 3 na DRAM (`a`, `b`, `c`) + 3 na L1 (cópias de trabalho), página = 4 bytes |
| Validação | `a + b` no host, com a mesma volta em 2³² do `uint32_t` |

O kernel tem três partes:

1. **leitura**: duas `noc_async_read_page` e uma única barreira;
2. **soma**: a L1 está mapeada na memória do RISC-V, então basta um ponteiro
   (`volatile tt_l1_ptr uint32_t*`). Depois do store, `ckernel::load_blocking`
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
./scripts/run.sh 01_riscv_add                 # 14 + 7
./scripts/run.sh 01_riscv_add 0xFFFFFFFF 1    # volta em 2^32 -> 0
TT_METAL_DPRINT_CORES=0,0 ./scripts/run.sh 01_riscv_add 100 23
```

Com `TT_METAL_DPRINT_CORES=0,0`, a mensagem do `DPRINT` do kernel aparece
(funciona no simulador):

```
Exemplo 01: 100 + 23 no RISC-V de movimentação de dados
Dispositivo 0 aberto: grade de núcleos Tensix 8x9 (72 núcleos)
0:0-0:BR: RISC-V: 100 + 23 = 123
Resultado: 123 (esperado 123)
PASSOU
```

Próximo passo: [02_eltwise_add](../02_eltwise_add) — a soma feita pelo
kernel de computação.
