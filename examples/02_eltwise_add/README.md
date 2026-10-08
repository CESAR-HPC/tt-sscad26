# 02 — Soma elemento a elemento (1 núcleo)

O menor programa que mostra o modelo de programação completo do
TT-Metalium:

```
host -> DRAM -> reader -> CB -> compute -> CB -> writer -> DRAM -> host
```

Seções 3.1–3.4 do capítulo.

## Operação e decomposição em tiles

`C = A + B`, com `A, B, C ∈ BF16^{M×N}` (padrão `M = N = 64`). As matrizes são
completadas com zeros até múltiplos de 32 e convertidas para o layout em
tiles (32x32, 4 faces de 16x16) no host (`common/layout.hpp`). Com `64x64`
são 4 tiles; o tile `i` de C depende apenas do tile `i` de A e de B.

## Mapeamento

| Item | Valor |
|---|---|
| Núcleos | 1 (`{0,0}`) |
| Reader | RISC-V 0, `kernels/dataflow/reader.cpp`: lê os tiles de A e B para `c_0` e `c_1` |
| Compute | `kernels/compute/compute.cpp`: `add_tiles` na FPU, resultado em `c_16` |
| Writer | RISC-V 1, `kernels/dataflow/writer.cpp`: escreve `c_16` na DRAM |
| CBs | `c_0` (A), `c_1` (B), `c_16` (C), 2 tiles cada (double buffering) |
| Validação | referência em CPU (float), tolerância de bfloat16 + PCC |

## Pontos de atenção

- **Circular buffers** são filas de tiles na L1. O produtor faz
  `cb_reserve_back` → escreve → `cb_push_back`; o consumidor faz
  `cb_wait_front` → lê → `cb_pop_front`. Com 2 tiles por CB, leitura, soma e
  escrita se sobrepõem.
- **Registrador Dst**: `tile_regs_acquire/commit` (MATH) e
  `tile_regs_wait/release` (PACK) formam o handshake entre a soma e o
  empacotamento do resultado.
- **Writer**: `noc_async_writes_flushed` por tile (basta o dado ter saído da
  L1 para liberar o CB) e uma única `noc_async_write_barrier` no fim.
- **Depuração**: `SSCAD_DPRINT=1 TT_METAL_DPRINT_CORES=0,0` compila e mostra
  um `DPRINT` do reader (desligado por padrão).

## Como executar

```bash
./scripts/run.sh 02_eltwise_add            # 64x64
./scripts/run.sh 02_eltwise_add 128 128
./scripts/run.sh 02_eltwise_add 50 70      # dimensões não múltiplas de 32
SSCAD_DPRINT=1 TT_METAL_DPRINT_CORES=0,0 ./scripts/run.sh 02_eltwise_add 32 64
```

Saída esperada:

```
Exemplo 02: C = A + B com A, B, C bfloat16 de 64x64 em 1 núcleo
Dispositivo 0 aberto: grade de núcleos Tensix 8x9 (72 núcleos)
Erro absoluto máximo: 3.906e-03 | erro relativo máximo: 1.969e-03 | PCC: 0.999999
PASSOU
```

Próximo passo: [03_eltwise_multicore](../03_eltwise_multicore).
