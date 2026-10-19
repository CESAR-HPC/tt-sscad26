# tt-sscad26

Código do minicurso **"Arquitetura Tenstorrent e programação com
TT-Metalium"** (SSCAD 2026). As listagens do capítulo são extraídas
diretamente deste repositório (regiões marcadas com `// ANCHOR: nome`).

Tudo roda no **simulador ttsim** (Wormhole), sem hardware.

## Início rápido (Docker)

```bash
docker compose up -d
docker compose exec sscad bash
./scripts/build.sh
./scripts/run_all.sh
```

### Sem Docker

Com um tt-metal v0.77.0 compilado (`build_Release`) e o `.so` do ttsim:

```bash
export TT_METAL_HOME=~/tt-metal
export TT_METAL_SIMULATOR=~/sim/libttsim_wh_x2.so   # caminho do .so do ttsim
./scripts/build.sh
./scripts/run_all.sh
```

`scripts/env.sh` completa as demais variáveis (`TT_METAL_RUNTIME_ROOT`,
`ARCH_NAME=wormhole_b0`, `TT_METAL_SLOW_DISPATCH_MODE=1`) sem sobrescrever
as que já estiverem definidas.

Um exemplo isolado: `./scripts/run.sh 02_eltwise_add`. Os exemplos não
recebem argumentos: tamanhos e valores são constantes no próprio `main.cpp`.

## Exemplos

| Exemplo | Conteúdo | Situação |
|---|---|---|
| [00_dram_loopback](examples/00_dram_loopback) | Movimentação de dados: DRAM → L1 → DRAM pela NoC | implementado |
| [01_riscv_add](examples/01_riscv_add) | O RISC-V de movimentação de dados como CPU, e seus limites | implementado |
| [02_eltwise_add](examples/02_eltwise_add) | `C = A + B`: reader → compute → writer, 1 núcleo | implementado |
| [03_eltwise_multicore](examples/03_eltwise_multicore) | O exemplo 02 dividido entre os núcleos | implementado |
| [04_gemv](examples/04_gemv) | `y = αAx + βy` | implementado |
| [05_gemm_v1_naive](examples/05_gemm_v1_naive) | GEMM ingênuo | implementado |
| [06_gemm_v2_pipelined](examples/06_gemm_v2_pipelined) | GEMM com leituras sobrepostas | implementado |
| [07_gemm_v3_reuse](examples/07_gemm_v3_reuse) | GEMM com reuso em L1 | implementado |
| [08_gemm_v4_mcast](examples/08_gemm_v4_mcast) | GEMM com multicast | implementado |
| [09_linear_gelu](examples/09_linear_gelu) | `Y = GELU(XW + b)` | implementado |

Cada exemplo tem `main.cpp` (host), `kernels/` e um README com o objetivo, o
mapeamento em núcleos e CBs e a saída esperada. Todas as chamadas ao
TT-Metalium ficam explícitas no `main.cpp` de cada exemplo, mesmo que isso
repita código entre eles, e as matrizes têm dimensões múltiplas de 32 (um
número exato de tiles). Os stubs têm um arquivo
`STUB` e são ignorados por `run_all.sh`.

## Estrutura

```
common/     utilitários de host sem TT-Metalium (referência em CPU, comparação, timer)
examples/   um diretório por exemplo
scripts/    env.sh, build.sh, run.sh, run_all.sh, check_kernels.sh
cmake/      alvo sscad_tt_metal e a função sscad_add_example
bench/      coleta de resultados (futuro)
```
