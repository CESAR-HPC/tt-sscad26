# 09 — Camada linear + GELU

> **Stub.** Este exemplo ainda não foi implementado: `main.cpp` apenas imprime
> uma mensagem e os kernels estão vazios. `scripts/run_all.sh` o ignora por
> causa do arquivo `STUB`.

## Objetivo

`Y = GELU(XW + b)`: GEMM v3 + soma do bias + GELU na SFPU, fundidos no kernel de computação.

Referência: `api/compute/eltwise_unary/gelu.h`

## O que cada kernel deve fazer

- **reader** (`kernels/dataflow/reader.cpp`): Como no GEMM v3, lendo também os tiles do bias b.
- **compute** (`kernels/compute/compute.cpp`): Após acumular XW no Dst, somar o bias e aplicar gelu_tile na SFPU antes de empacotar.
- **writer** (`kernels/dataflow/writer.cpp`): Escrever os tiles de Y.
