# 04 — GEMV multicore

`y = αAx + βy`, com `A` e `x` em bfloat16 e `y`, `α` e `β` em float (FP32).
O produto `Ax` é acumulado em FP32 no registrador Dst.

## Vetores como matrizes de tiles (padding)

O motor de matrizes (FPU) só multiplica tiles 32x32, então `x` e `y` precisam
virar matrizes. O preenchimento é feito no próprio `main.cpp` (âncora
`host_padding`):

| Vetor | Matriz | Conteúdo |
|---|---|---|
| `x` (n) | `n x m` | `x[i]` na coluna 0 de cada coluna de tiles (`padded[i][32 j] = x[i]`), zero no resto |
| `y` (m) | `m x m` | `y[i]` na coluna 0 (`padded[i][0] = y[i]`), zero no resto |
| `α`, `β` | `32 x 32` | um tile cheio com o valor do escalar |

Com isso, `A(mt, nt) · X(nt, mt)` deixa na coluna 0 do tile de saída a
contribuição do bloco `nt` para `y[32 mt : 32 mt + 32]`, e as demais colunas
ficam zeradas. O resultado volta no formato de `y`: o host lê `y[i]` da
coluna 0 (âncora `host_unpad`).

Este é o único exemplo com código de preenchimento: as dimensões continuam
múltiplas de 32, mas os vetores têm uma só coluna.

## Divisão do trabalho

A unidade de trabalho é uma **linha de tiles** de `A` (32 linhas de `A`, um
tile de `y`). `split_work_to_cores(núcleos, Mt)` distribui as `Mt` linhas de
tiles entre os núcleos, como no exemplo 03.

Para a linha de tiles `mt`:

| Kernel | O que faz |
|---|---|
| reader | lê `α` e `β` uma vez; depois, o tile `(mt, 0)` de `y` e os pares `A(mt, nt)`, `X(nt, mt)` para `nt = 0..Nt-1` (uma barreira por par) |
| compute | fase 1 (FPU): `Dst = Σ A(mt, nt) · X(nt, mt)` com `matmul_tiles` → `c_5`; fase 2 (SFPU): `α·(Ax) + β·y` com `mul_binary_tile`/`add_binary_tile` → `c_16` |
| writer | escreve o tile de saída na posição `(mt, 0)` |

As duas fases do compute leem formatos diferentes (bfloat16 no matmul, float
na fase 2). O unpacker não se reconfigura sozinho, então cada fase começa com
`reconfig_data_format`/`reconfig_data_format_srca`.

## Mapeamento

| Item | Valor |
|---|---|
| Operação | `y = 2·Ax + 0.5·y`, `A` de `512x256` (`Mt = 16`, `Nt = 8`) |
| Núcleos | 16 (uma linha de tiles por núcleo) |
| CBs | `c_0` (A) e `c_1` (x): bfloat16, 2 tiles; `c_2` (y) e `c_16` (saída): float, 2 tiles; `c_3` (α), `c_4` (β) e `c_5` (Ax): float, 1 tile |
| Compute | `HiFi4`, `fp32_dest_acc_en = true` |
| Validação | `reference::gemv` em CPU, tolerância de bfloat16 |

## Como executar

```bash
./scripts/run.sh 04_gemv
```

Saída esperada:

```
Exemplo 04: y = 2 * A x + 0.5 * y com A 512x256 bfloat16 e y float em 16 de 8x9 núcleos
Erro absoluto máximo: 1.760e-02 | erro relativo máximo: 5.260e-04 | PCC: 1.000000
PASSOU
```

Próximo: o [exemplo 05](../05_gemm_v1_naive) troca o vetor `x` por uma matriz
`B` e calcula `C = AB`.
