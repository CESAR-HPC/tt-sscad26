# 08 — GEMM v4 (multicast) — meta estendida

> **Stub.** Este exemplo ainda não foi implementado: `main.cpp` apenas imprime
> uma mensagem e os kernels estão vazios. `scripts/run_all.sh` o ignora por
> causa do arquivo `STUB`.

## Objetivo

GEMM v3 + multicast de A ao longo das linhas e de B ao longo das colunas da grade, sincronizado com semáforos.

Referência: `programming_examples/matmul/matmul_multicore_reuse_mcast`

## O que cada kernel deve fazer

- **reader** (`kernels/dataflow/reader.cpp`): Núcleos da primeira coluna/linha leem A/B da DRAM e fazem multicast para a linha/coluna; os demais esperam no semáforo.
- **compute** (`kernels/compute/compute.cpp`): Igual ao v3.
- **writer** (`kernels/dataflow/writer.cpp`): Igual ao v3.

## Saída para o benchmark

Os exemplos GEMM (05–08) não recebem argumentos de linha de comando: os
tamanhos são constantes no `main.cpp`, múltiplos de 32. Cada um deve imprimir
uma linha CSV para o benchmark:

```
example,m,n,k,dtype,cores,time_ms,gflops,max_rel_err
```
