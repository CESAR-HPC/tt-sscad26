# 04 — GEMV multicore

> **Stub.** Este exemplo ainda não foi implementado: `main.cpp` apenas imprime
> uma mensagem e os kernels estão vazios. `scripts/run_all.sh` o ignora por
> causa do arquivo `STUB`.

## Objetivo

Calcular `y = αAx + βy` com entradas bfloat16 e acumulação em FP32, distribuindo as linhas de tiles de `A` entre os núcleos.

Referência: `petro-tt/tt-blas/blas/gemv`

## O que cada kernel deve fazer

- **reader** (`kernels/dataflow/reader.cpp`): Ler, para cada linha de tiles de A atribuída ao núcleo, os tiles de A e de x (e uma vez os tiles de α, β e y).
- **compute** (`kernels/compute/compute.cpp`): Para cada linha de tiles: acumular A·x no registrador Dst (matmul_tiles), depois aplicar α e somar β·y; empacotar o tile de y.
- **writer** (`kernels/dataflow/writer.cpp`): Escrever os tiles de y da faixa do núcleo na DRAM.
