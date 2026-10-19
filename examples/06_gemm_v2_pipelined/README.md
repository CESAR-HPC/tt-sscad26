# 06 — GEMM v2 (pipeline)

`C = AB`, com os mesmos tipos, tamanhos e divisão de trabalho do exemplo 05
(`A` e `B` em bfloat16, `C` em float, acumulação em FP32 no Dst). Muda só a
movimentação de dados: as leituras de cada tile de saída são emitidas juntas,
sob uma única barreira, e os CBs comportam o tile de saída atual e o próximo,
para que leitura e cálculo se sobreponham.

## O que muda em relação ao v1

| | v1 (05) | v2 (06) |
|---|---|---|
| CBs de A e B | 1 tile | `2·Kt` tiles (linha de A / coluna de B de dois tiles de saída) |
| CB de C | 1 tile | 2 tiles |
| Leituras por barreira | 1 | `2·Kt` (todo o par linha/coluna de um tile de saída) |
| Writer | barreira por tile | `async_writes_flushed` por tile, uma barreira no fim |
| Compute | — | idêntico |

O compute continua consumindo um tile de cada vez (`cb_wait_front(…, 1)`),
embora o reader entregue `Kt` tiles por `push_back`: produtor e consumidor de
um CB não precisam usar a mesma granularidade.

O reuso continua nulo: cada tile de `A` ainda é lido `Nt` vezes e cada tile
de `B`, `Mt` vezes. Isso é o que o exemplo 07 resolve.

## Mapeamento

| Item | Valor |
|---|---|
| Operação | `C = AB`, `m = 576`, `n = 512`, `k = 384` (`Mt = 18`, `Nt = 16`, `Kt = 12`) |
| Núcleos | a grade inteira (72 no simulador): 288 tiles → 4 por núcleo |
| CBs | `c_0` (A) e `c_1` (B): bfloat16, `2·Kt = 24` tiles; `c_16` (C): float, 2 tiles |
| Compute | `HiFi4`, `fp32_dest_acc_en = true` |
| Validação | `reference::gemm` em CPU, tolerância de bfloat16 |

Com `k` grande, `2·Kt` tiles por CB deixam de caber na L1; aí é preciso
percorrer `K` em blocos, como no exemplo 07.

## Saída para o benchmark

O programa executa o workload duas vezes (a primeira inclui a compilação JIT
dos kernels) e mede a segunda. Depois imprime uma linha CSV:

```
example,m,n,k,dtype,cores,time_ms,gflops,max_rel_err
```

`gflops = 2mnk / tempo`. No simulador o tempo não representa o hardware.

## Como executar

```bash
./scripts/run.sh 06_gemm_v2_pipelined
```

Saída esperada (o tempo varia):

```
Exemplo 06: C = A B com A 576x384, B 384x512 bfloat16 e C float em 72 de 8x9 núcleos
  72 núcleos com 4 tiles + 0 núcleos com 0 tiles
Erro absoluto máximo: 5.563e-03 | erro relativo máximo: 1.879e-04 | PCC: 1.000000
example,m,n,k,dtype,cores,time_ms,gflops,max_rel_err
06_gemm_v2_pipelined,576,512,384,bf16-fp32,72,1165.598,0.194,1.879e-04
PASSOU
```

O resultado é idêntico ao do exemplo 05: a ordem das somas no Dst é a mesma.

Próximo: o exemplo 07 reaproveita em L1 os tiles de A e B.
