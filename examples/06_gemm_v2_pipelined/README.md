# 06 — GEMM v2 (pipeline)

> **Stub.** Este exemplo ainda não foi implementado: `main.cpp` apenas imprime
> uma mensagem e os kernels estão vazios. `scripts/run_all.sh` o ignora por
> causa do arquivo `STUB`.

## Objetivo

GEMM v1 com CBs de 2 ou mais tiles e leituras de A/B agrupadas sob uma única barreira, sobrepondo a latência das leituras com a computação.

Referência: —

## O que cada kernel deve fazer

- **reader** (`kernels/dataflow/reader.cpp`): Como no v1, mas reservando vários tiles por vez e emitindo várias leituras antes de uma única barreira.
- **compute** (`kernels/compute/compute.cpp`): Igual ao v1.
- **writer** (`kernels/dataflow/writer.cpp`): Igual ao v1, com noc_async_writes_flushed por tile e uma barreira final.

## Interface comum (exemplos GEMM)

Todos os exemplos GEMM (05–08) devem aceitar a mesma linha de comando:

```
--m <M> --n <N> --k <K> --iters <repetições> --check
```

e imprimir uma linha CSV para o benchmark:

```
example,m,n,k,dtype,cores,time_ms,gflops,max_rel_err
```
