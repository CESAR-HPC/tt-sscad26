# 05 — GEMM v1 (ingênuo)

`C = AB`, com `A` (`m x k`) e `B` (`k x n`) em bfloat16 e `C` em float, com
acumulação em FP32 no Dst. Porte do GEMM do tt-blas com `TensorAccessor`: um
tile de saída por iteração, uma barreira por tile lido e nenhum reuso de
dados. É a versão de referência que os exemplos 06–08 otimizam.

Referência: `petro-tt/tt-blas/blas/gemm` (sem `α` e `β`) e
`tt-metal/tt_metal/programming_examples/matmul/matmul_multi_core`.

## Decomposição em tiles

O tile `(i, j)` de `C` é `Σ_kt A(i, kt) · B(kt, j)`, para `kt = 0..Kt-1`. Os
`Mt·Nt` tiles de `C` são numerados em row-major (`t = i·Nt + j`) e divididos
entre os núcleos com `split_work_to_cores`, como no exemplo 03.

| Kernel | O que faz | Argumentos de execução |
|---|---|---|
| reader | para cada tile `t` da faixa: lê `A(i, kt)` e `B(kt, j)`, `kt = 0..Kt-1`, **uma barreira por tile** | endereços de A e B, `Kt`, `Nt`, `start_tile`, `num_tiles` |
| compute | `tile_regs_acquire` (zera o Dst), `Kt` × `matmul_tiles`, `pack_tile` | `Kt`, `num_tiles` |
| writer | escreve o tile `t` de C, uma barreira por tile | endereço de C, `start_tile`, `num_tiles` |

## Por que é ingênuo

- **CBs de 1 tile.** O reader só lê o próximo tile depois que o compute
  consumiu o anterior: leitura e cálculo nunca se sobrepõem.
- **Uma barreira por leitura.** O reader espera a DRAM a cada tile, em vez de
  emitir várias leituras e esperar uma vez (exemplo 06).
- **Nenhum reuso.** Cada tile de `A` é lido `Nt` vezes e cada tile de `B`,
  `Mt` vezes, da DRAM (exemplo 07 mantém blocos em L1; o 08 usa multicast).

## Mapeamento

| Item | Valor |
|---|---|
| Operação | `C = AB`, `m = 256`, `n = 320`, `k = 192` (`Mt = 8`, `Nt = 10`, `Kt = 6`) |
| Núcleos | a grade inteira (72 no simulador): 80 tiles → 8 núcleos com 2 e 64 com 1 |
| CBs | `c_0` (A) e `c_1` (B): bfloat16, 1 tile; `c_16` (C): float, 1 tile |
| Compute | `HiFi4`, `fp32_dest_acc_en = true` |
| Validação | `reference::gemm` em CPU, tolerância de bfloat16 |

## Saída para o benchmark

O programa executa o workload duas vezes (a primeira inclui a compilação JIT
dos kernels) e mede a segunda. Depois imprime uma linha CSV:

```
example,m,n,k,dtype,cores,time_ms,gflops,max_rel_err
```

`gflops = 2mnk / tempo`. No simulador o tempo não representa o hardware.

## Como executar

```bash
./scripts/run.sh 05_gemm_v1_naive
```

Saída esperada (o tempo varia):

```
Exemplo 05: C = A B com A 256x192, B 192x320 bfloat16 e C float em 72 de 8x9 núcleos
  8 núcleos com 2 tiles + 64 núcleos com 1 tiles
Erro absoluto máximo: 3.836e-03 | erro relativo máximo: 1.818e-04 | PCC: 1.000000
example,m,n,k,dtype,cores,time_ms,gflops,max_rel_err
05_gemm_v1_naive,256,320,192,bf16-fp32,72,430.331,0.073,1.818e-04
PASSOU
```

Próximo: o exemplo 06 sobrepõe leituras e cálculo.
