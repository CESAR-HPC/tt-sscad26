# 05 — GEMM v1 (ingênuo)

> **Stub.** Este exemplo ainda não foi implementado: `main.cpp` apenas imprime
> uma mensagem e os kernels estão vazios. `scripts/run_all.sh` o ignora por
> causa do arquivo `STUB`.

## Objetivo

Porte fiel do GEMM do tt-blas: um tile de saída por iteração, uma barreira por tile lido e nenhum reuso de dados, usando `TensorAccessor`.

Referência: `petro-tt/tt-blas/blas/gemm`

## O que cada kernel deve fazer

- **reader** (`kernels/dataflow/reader.cpp`): Para cada tile de saída (i, j): ler os Kt tiles da linha i de A e da coluna j de B, com uma barreira por tile.
- **compute** (`kernels/compute/compute.cpp`): Para cada tile de saída: acumular Kt produtos matmul_tiles no Dst e empacotar o resultado.
- **writer** (`kernels/dataflow/writer.cpp`): Escrever cada tile de C na DRAM.

## Saída para o benchmark

Os exemplos GEMM (05–08) não recebem argumentos de linha de comando: os
tamanhos são constantes no `main.cpp`, múltiplos de 32. Cada um deve imprimir
uma linha CSV para o benchmark:

```
example,m,n,k,dtype,cores,time_ms,gflops,max_rel_err
```
