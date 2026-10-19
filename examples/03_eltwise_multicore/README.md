# 03 — Soma elemento a elemento (vários núcleos)

O exemplo 02 dividido entre vários núcleos Tensix. Os kernels são
praticamente os mesmos; mudam apenas a divisão do trabalho no host e o
índice do primeiro tile passado ao reader e ao writer.

## Divisão do trabalho

`split_work_to_cores(núcleos, num_tiles)` devolve dois grupos de núcleos:
`group_1`, em que cada núcleo processa `tiles_per_core_1` tiles, e
`group_2`, com um tile a menos por núcleo (vazio se a divisão for exata). O
host percorre os dois grupos na mesma ordem, atribuindo a cada núcleo a faixa
contígua `[start_tile, start_tile + tiles_per_core)`:

| Kernel | Argumentos de execução |
|---|---|
| reader | endereço de A, endereço de B, `start_tile`, `num_tiles` |
| compute | `num_tiles` (só enxerga os CBs) |
| writer | endereço de C, `start_tile`, `num_tiles` |

CBs e kernels são criados de uma vez em todos os núcleos, passando o
`CoreRangeSet` em vez de um único `CoreCoord`.

## Mapeamento

| Item | Valor |
|---|---|
| Operação | `C = A + B`, bfloat16, `512x512` (256 tiles) |
| Núcleos | a grade inteira (8x9 = 72 no simulador) |
| CBs | `c_0`, `c_1`, `c_16`, 2 tiles cada |
| Validação | igual ao exemplo 02 |

## Como executar

```bash
./scripts/run.sh 03_eltwise_multicore
```

Os 256 tiles não se dividem igualmente pelos 72 núcleos, então aparecem os
dois grupos: 40 núcleos com 4 tiles e 32 com 3.

Saída esperada:

```
Exemplo 03: C = A + B com 512x512 bfloat16 (256 tiles) em 72 de 8x9 núcleos
  40 núcleos com 4 tiles + 32 núcleos com 3 tiles
Erro absoluto máximo: 4.395e-03 | erro relativo máximo: 2.197e-03 | PCC: 0.999999
PASSOU
```
