# 07 — GEMM v3 (reuso em L1)

> **Stub.** Este exemplo ainda não foi implementado: `main.cpp` apenas imprime
> uma mensagem e os kernels estão vazios. `scripts/run_all.sh` o ignora por
> causa do arquivo `STUB`.

## Objetivo

Cada núcleo calcula um bloco de saída; blocos de A e B são mantidos na L1 e reutilizados (`in0_block_w`, `out_subblock_h/w`).

Referência: `programming_examples/matmul/matmul_multicore_reuse`

## O que cada kernel deve fazer

- **reader** (`kernels/dataflow/reader.cpp`): Ler blocos de A (out_block_h x in0_block_w) e de B (in0_block_w x out_block_w) para cada passo em K.
- **compute** (`kernels/compute/compute.cpp`): Acumular sub-blocos de saída (out_subblock_h x out_subblock_w) no Dst reutilizando os blocos de A e B na L1.
- **writer** (`kernels/dataflow/writer.cpp`): Escrever o bloco de saída do núcleo.

## Interface comum (exemplos GEMM)

Todos os exemplos GEMM (05–08) devem aceitar a mesma linha de comando:

```
--m <M> --n <N> --k <K> --iters <repetições> --check
```

e imprimir uma linha CSV para o benchmark:

```
example,m,n,k,dtype,cores,time_ms,gflops,max_rel_err
```
