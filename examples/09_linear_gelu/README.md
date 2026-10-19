# 09 — Camada linear + GELU

`Y = GELU(XW + b)`, com `X` (`m x k`), `W` (`k x n`) e o bias `b` (`n`) em
bfloat16 e `Y` em float, acumulação em FP32. É a camada linear seguida da
ativação de um MLP de transformer. O GEMM é o do exemplo 07 (GEMM v3, blocos
reaproveitados em L1); a soma do bias e a GELU são **fundidas** no kernel de
computação, de modo que `XW` e `XW + b` nunca vão para a DRAM.

## O bias em tiles

A FPU trabalha com tiles de 32 x 32. O host monta o bias como uma matriz
`32 x n` com `b` na linha 0 e zeros no resto, ou seja, `Nt` tiles com o bias
na primeira linha. No compute, `add_tiles_bcast_rows` repete essa linha nas 32
linhas do tile de `XW`: `(XW)(i, j) + b(j)` para toda linha `i`.

## Kernels

| Kernel | O que faz |
|---|---|
| reader | lê uma vez os `per_core_N` tiles de bias do bloco; depois, como no exemplo 07, um bloco de `X` e um de `W` por passo em `K` |
| compute, fase 1 (FPU) | GEMM do exemplo 07, mas o último passo também empacota em `c_24`: ao fim, `c_24 = XW` |
| compute, fase 2 (FPU + SFPU) | por sub-bloco: `Dst = c_24 + bias` (`add_tiles_bcast_rows`), `Dst = GELU(Dst)` (`gelu_tile`), empacota em `c_16` |
| writer | idêntico ao do exemplo 07 |

A soma com broadcast lê os dois operandos de CBs (SrcA e SrcB), e não do Dst:
por isso `XW` passa por `c_24` antes da fase 2. A GELU, na SFPU, opera direto
sobre o Dst, logo depois da soma, sem outro empacotamento.

As fases leem formatos diferentes (`X`, `W` e bias em bfloat16; `c_24` em
float), então cada troca de operandos começa com uma reconfiguração do
unpacker (`copy_tile_to_dst_init_short_with_dt`, `reconfig_data_format_srca`,
`reconfig_data_format`), como no exemplo 04.

## Mapeamento

| Item | Valor |
|---|---|
| Operação | `Y = GELU(XW + b)`, `m = 576`, `n = 512`, `k = 384` (`Mt = 18`, `Nt = 16`, `Kt = 12`) |
| Blocos | os do exemplo 07: `2 x 2` tiles por núcleo, `in0_block_w = 2`, sub-blocos `1 x 2` |
| Núcleos | 72 (grade de 8 x 9 blocos) |
| CBs | `c_0` (X), `c_1` (W): bfloat16, 2 blocos; `c_2` (bias): bfloat16, `per_core_N` tiles; `c_16` (Y) e `c_24` (parciais): float, o bloco inteiro |
| Compute | `HiFi4`, `fp32_dest_acc_en = true`; GELU exata (`gelu_tile<false>`) |
| Validação | `reference::linear_gelu` em CPU (GELU com `erf`), tolerância de bfloat16 |

## Saída para o benchmark

O programa executa o workload duas vezes (a primeira inclui a compilação JIT
dos kernels) e mede a segunda. Depois imprime uma linha CSV:

```
example,m,n,k,dtype,cores,time_ms,gflops,max_rel_err
```

`gflops` conta só as `2mnk` operações do GEMM; o bias e a GELU são `O(mn)`.
No simulador o tempo não representa o hardware.

## Como executar

```bash
./scripts/run.sh 09_linear_gelu
```

Saída esperada (o tempo varia):

```
Exemplo 09: Y = GELU(X W + b) com X 576x384, W 384x512 e b 512 bfloat16 e Y float em 72 de 8x9 núcleos
Erro absoluto máximo: 5.868e-02 | erro relativo máximo: 1.976e-03 | PCC: 1.000000
example,m,n,k,dtype,cores,time_ms,gflops,max_rel_err
09_linear_gelu,576,512,384,bf16-fp32,72,1361.715,0.166,1.976e-03
PASSOU
```

O erro é um pouco maior que o do exemplo 07: `XW` passa mais uma vez por SrcA
(TF32) na soma do bias, e a GELU da SFPU é uma aproximação polinomial.
