# SKILLS — tt-sscad26 (code for the SSCAD 2026 short course)

## Scope

This repository holds the example code for the SSCAD 2026 short course and book
chapter on Tenstorrent / **TT-Metalium**. The examples are **teaching
material**: clarity beats cleverness, and the chapter (written in Portuguese)
quotes them directly.

Examples, from simple to complex:
`00_dram_loopback` → `01_riscv_add` → `02_eltwise_add` → `03_eltwise_multicore` → `04_gemv` → `05–08` GEMM (naive → pipelined → L1 reuse → multicast) → `09_linear_gelu`.

The two introductory examples have a specific teaching role; keep them small:

- `00_dram_loopback` — **how data moves**: one data movement kernel copies tiles DRAM → L1 → DRAM over the NoC (asynchronous transfers + barriers). No compute kernel, no CBs.
- `01_riscv_add` — **the data movement core is a CPU**: a "baby" RISC-V (RV32IM, no FPU, scalar, ~1 GHz, KiB-sized local memory) adds two `uint32` in plain C++. Its point is the *limitations*: use these cores for control and addressing, and leave arithmetic on data to the compute kernel (`02`). Cite `~/tt-isa-documentation/WormholeB0/TensixTile/BabyRISCV/` for hardware facts.

Upstream references: `programming_examples/loopback` and `programming_examples/add_2_integers_in_riscv` — the latter uses the deprecated `InterleavedAddrGen`, so port its DRAM access to `TensorAccessor`.

Everything runs in **Docker with the ttsim simulator** (Wormhole), so no hardware is needed.
Real hardware is used only for the performance measurements.

The skeleton and the conventions are specified in `SPEC_SKELETON.md`, which takes precedence over this file.

## Related repositories

| Repo | What to use it for |
|---|---|
| `~/cesar/petro-tt` — **only `tt-blas/blas/`** | The BLAS implementation that inspires the examples (GEMV, GEMM, benchmark harness, layout utils). Its kernels use the legacy API; branch `refactor/workshop-api` shows the current API. Copy ideas, not the legacy API. |
| `~/tt-metal` (v0.77.0) | Framework source and examples (`tt_metal/programming_examples/`: `vecadd_multi_core`, `matmul/*`), `METALIUM_GUIDE.md`, `tech_reports/`. Source of truth for API signatures. |
| `~/tt-isa-documentation` | Hardware/ISA reference (`WormholeB0/`, `BlackholeA0/`). Use it when a design choice depends on hardware details (L1 size, NoC, Tensix engines). |
| Chapter repo (`.../SBCbookchapter Folder`) | `book_chapter_v2.tex` says which example backs which section. |

## Rules

1. **TT-Metalium v0.77.0 only.** Host: `MeshDevice`, `MeshBuffer`, `CreateKernel`, `SetRuntimeArgs`. Kernels: `TensorAccessor`. **Never** use `InterleavedAddrGenFast`/`InterleavedAddrGen`, `noc_async_read_tile`/`noc_async_write_tile` or `*_tiles_init`. Don't use the experimental `ProgramSpec` API.
2. **Check signatures in `~/tt-metal`** before using an API; don't guess.
3. **Identifiers in English; comments, READMEs and console output in Portuguese.**
4. **Every example must validate** against a CPU reference and print `PASSOU`/`FALHOU`. `./scripts/run_all.sh` must stay green.
5. Mark code the chapter quotes with `// ANCHOR: name` / `// ANCHOR_END: name`. Don't rename existing anchors.
6. Keep each example self-contained (`main.cpp` + `kernels/{dataflow,compute}/`) and its README up to date.
7. Small commits on feature branches; **don't push or open PRs without the owner's approval**.
