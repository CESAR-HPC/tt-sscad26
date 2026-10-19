# 08 — GEMM v4 (multicast)

`C = AB`, com os tipos, tamanhos e blocos do exemplo 07. O compute e o writer
são os mesmos; o reader passa a usar **multicast** pela NoC: cada bloco de
`A` é lido da DRAM por um único núcleo da linha da grade e enviado aos demais
da linha, e cada bloco de `B`, por um único núcleo da coluna.

Um único kernel de reader deduz o seu papel (emissor ou receptor de `A` e de
`B`) da posição do bloco.

## Papéis dos núcleos

A grade tem `X = Nt/per_core_N = 8` colunas e `Y = Mt/per_core_M = 9` linhas;
o bloco `(by, bx)` está no núcleo lógico `(bx, by)`.

| Núcleo | A | B |
|---|---|---|
| `(0, 0)` | lê e envia para `(1..X-1, 0)` | lê e envia para `(0, 1..Y-1)` |
| `(0, y)`, `y > 0` | lê e envia para `(1..X-1, y)` | recebe de `(0, 0)` |
| `(x, 0)`, `x > 0` | recebe de `(0, 0)` | lê e envia para `(x, 1..Y-1)` |
| `(x, y)`, `x, y > 0` | recebe de `(0, y)` | recebe de `(x, 0)` |

Cada bloco de `A` sai da DRAM uma vez (no exemplo 07, `X` vezes) e cada bloco
de `B` uma vez (no 07, `Y` vezes).

## Sincronização

Quatro semáforos em L1, criados com `CreateSemaphore` em todos os núcleos
(`a_ready`, `a_valid`, `b_ready`, `b_valid`). Para cada bloco:

1. **receptor**: `reserve_back` no CB, marca o próprio `*_valid` como
   `INVALID`, incrementa o `*_ready` do emissor (`Semaphore::up`, uma operação
   atômica remota) e espera `*_valid == VALID`;
2. **emissor**: lê o bloco da DRAM, espera `*_ready == receptores` (todos
   têm espaço no CB), zera `*_ready`, envia o bloco com
   `Noc::async_write_multicast` e então envia `VALID` aos `*_valid` dos
   receptores com `Semaphore::set_multicast`. Na mesma NoC o semáforo chega
   depois dos dados (no Blackhole é preciso um `async_writes_flushed` entre os
   dois).

O bloco é escrito no mesmo endereço do CB em todos os núcleos: os CBs têm a
mesma configuração e avançam em sincronia.

## NoCs e coordenadas

O reader usa a NoC 0 na coluna 0 e a NoC 1 nas demais, e o writer, a outra
NoC de cada núcleo. Os kernels endereçam os outros
núcleos por coordenadas da NoC (`worker_core_from_logical_core` no host). O
retângulo de um multicast vai do canto `start` ao `end` no sentido da NoC: na
NoC 0, das coordenadas menores para as maiores; na NoC 1, ao contrário. O host
passa o retângulo já na ordem certa para a NoC de cada reader.

## Mapeamento

| Item | Valor |
|---|---|
| Operação | `C = AB`, `m = 576`, `n = 512`, `k = 384` (`Mt = 18`, `Nt = 16`, `Kt = 12`) |
| Blocos | `per_core_M x per_core_N = 2 x 2`, `in0_block_w = 2`, sub-blocos `1 x 2` |
| Núcleos | 72 (grade de 8 x 9 blocos); cada multicast de A tem `X-1 = 7` receptores, e cada um de B, `Y-1 = 8` |
| CBs | os do exemplo 07 (`c_0`, `c_1`, `c_16`, `c_24`) |
| Compute | idêntico ao do exemplo 07 |
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
./scripts/run.sh 08_gemm_v4_mcast
```

Saída esperada (o tempo varia):

```
Exemplo 08: C = A B com A 576x384, B 384x512 bfloat16 e C float em 72 de 8x9 núcleos
  bloco de 2x2 tiles por núcleo, passo de 2 tiles em K, sub-blocos de 1x2
Erro absoluto máximo: 4.487e-02 | erro relativo máximo: 1.516e-03 | PCC: 1.000000
example,m,n,k,dtype,cores,time_ms,gflops,max_rel_err
08_gemm_v4_mcast,576,512,384,bf16-fp32,72,1210.482,0.187,1.516e-03
PASSOU
```

O resultado é idêntico ao do exemplo 07: só muda de onde vêm os blocos.

Próximo: o exemplo 09 funde bias e GELU ao GEMM.
