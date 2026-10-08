# SPEC — Skeleton of the `tt-sscad26` repository

## 1. Context

`tt-sscad26` holds the code that accompanies the SSCAD 2026 mini-course chapter
*"Arquitetura Tenstorrent e programação com TT-Metalium"* (written in Portuguese,
`book_chapter_v2.tex` in the Overleaf repo). Code listings in the chapter are
extracted **directly from this repository**, so code here is teaching material:
clarity beats cleverness, and every example must build and run inside the
simulator container with no hardware and no extra installation.

The examples are inspired by `~/cesar/petro-tt/tt-blas` (GEMV, GEMM, benchmark
harness, layout utilities) and by the official examples in
`~/tt-metal/tt_metal/programming_examples` (`loopback`, `add_2_integers_in_riscv`,
`vecadd_multi_core`, `matmul/*`).

This spec covers **only the skeleton** plus **four complete examples**:

- **00 `dram_loopback`** — introduction to data movement: one data movement
  kernel copies tiles DRAM → L1 → DRAM through the NoC. No compute kernel, no CBs.
- **01 `riscv_add`** — the data movement core is a small general-purpose
  ("baby") RISC-V CPU: it reads two integers from DRAM, adds them with plain C++
  and writes the sum back. Shows what that CPU can do and, above all, its limits.
- **02 `eltwise_add`** — the full reader → compute → writer pipeline; proves the
  toolchain end-to-end.
- **03 `eltwise_multicore`** — example 02 split across the core grid (§5.4).

The remaining examples get their directory, README and stub files, and are
implemented in later tasks.

## 2. Fixed decisions (do not change)

| Topic | Decision |
|---|---|
| TT-Metalium version | **v0.77.0** (same as petro-tt `Dockerfile.build`). |
| Environment | Docker image with TT-Metalium + **ttsim** (Wormhole simulator). The user only needs Docker. |
| Base image | `ghcr.io/cesar-hpc/petro-tt/cesar-wormhole-simulator-ubuntu-22.04-release-amd64` (the one used by `petro-tt/tt-blas/Dockerfile`). Pin a concrete tag if one exists for v0.77.0; otherwise use `latest-rc` and record the image digest in the README. |
| Host API | `distributed::MeshDevice`, `MeshBuffer`, `MeshWorkload`, `EnqueueMeshWorkload`, `EnqueueWrite/ReadMeshBuffer`; `Program` + `CreateCircularBuffer` + `CreateKernel` + `SetRuntimeArgs`. **Do not** use the experimental `experimental::ProgramSpec`/`MakeProgramFromSpec` (metal2) API. |
| Kernel API | **`TensorAccessor` / `TensorAccessorArgs`** for DRAM access. **Do not** use `InterleavedAddrGenFast`/`InterleavedAddrGen` (their `noc_async_read_page`/`write_page` overloads are `[[deprecated]]` in v0.77). Use the new compute names (`add_init`, `mul_init`, …) rather than the deprecated `*_tiles_init`. |
| Language | C++20. Identifiers in English; **code comments, READMEs and console messages in Portuguese (pt-BR)**, because the listings go into a Portuguese chapter. |
| Target arch | `ARCH_NAME=wormhole_b0`; simulator via `TT_METAL_SIMULATOR`; `TT_METAL_SLOW_DISPATCH_MODE=1` is required by the simulator. |
| License | Apache-2.0 (same as tt-metal). |

## 3. Target layout

```text
tt-sscad26/
├── README.md                  # pt-BR: what the repo is, quick start in 4 commands, example list
├── LICENSE
├── Dockerfile                 # FROM simulator image; WORKDIR /workdir; no code copied in (volume mount)
├── docker-compose.yml         # service "sscad", volume .:/workdir, env vars below, tail -f /dev/null
├── .dockerignore / .gitignore # ignore build/, results/*.csv, __pycache__
├── CMakeLists.txt             # top-level; add_subdirectory(common) + one per example
├── cmake/
│   └── tt_metal.cmake         # INTERFACE target `sscad_tt_metal` (includes, link dirs, libs, flags)
├── common/                    # header-only host-side utilities (namespace sscad)
│   ├── CMakeLists.txt
│   ├── device.hpp             # open/close MeshDevice 1x1, print grid size and arch
│   ├── layout.hpp             # row-major <-> tiled (32x32, faces) + padding to multiples of 32
│   ├── data_format.hpp        # C++ type -> tt::DataFormat (bfloat16, float)
│   ├── reference.hpp          # CPU references: eltwise add, gemv, gemm, linear+gelu
│   ├── compare.hpp            # max abs/rel error, PCC; returns pass/fail with tolerance
│   └── timer.hpp              # wall-clock timing for host + (optional) device markers
├── examples/
│   ├── 00_dram_loopback/      # COMPLETE in this spec (see §5.1)
│   ├── 01_riscv_add/          # COMPLETE in this spec (see §5.2)
│   ├── 02_eltwise_add/        # COMPLETE in this spec (see §5.3)
│   ├── 03_eltwise_multicore/  # COMPLETE in this spec (see §5.4)
│   ├── 04_gemv/               # stub
│   ├── 05_gemm_v1_naive/      # stub
│   ├── 06_gemm_v2_pipelined/  # stub
│   ├── 07_gemm_v3_reuse/      # stub
│   ├── 08_gemm_v4_mcast/      # stub (stretch goal)
│   └── 09_linear_gelu/        # stub
├── scripts/
│   ├── env.sh                 # exports the env vars (sourced inside the container)
│   ├── build.sh               # cmake -G Ninja -B build && cmake --build build
│   ├── run.sh <example> [args]# runs one example with the correct env
│   └── run_all.sh             # runs every *implemented* example; non-zero exit on any failure
└── bench/
    ├── README.md              # how results are collected (filled in later)
    └── results/.gitkeep
```

### Per-example layout (all 10)

```text
examples/NN_name/
├── CMakeLists.txt
├── README.md        # pt-BR: goal, mathematical operation, tile decomposition, core mapping,
│                    # CBs used, chapter section it backs, how to run, expected output
├── main.cpp         # host program
└── kernels/
    ├── dataflow/reader.cpp
    ├── dataflow/writer.cpp
    └── compute/compute.cpp
```

Examples 00 and 01 use **a single data movement kernel and no compute kernel**,
so they only have `kernels/dataflow/<name>.cpp` (`loopback.cpp` and
`riscv_add.cpp`); they have no `writer.cpp` or `compute/`. Every other example
follows the three-kernel layout above.

For stubs: `main.cpp` prints `"Exemplo NN ainda não implementado"` and returns 0;
the kernel files contain only `kernel_main() {}` and a header comment describing
what the kernel must do. **`run_all.sh` must skip stubs** (e.g. a `STUB` marker file
in the directory), so it stays green.

## 4. Infrastructure requirements

### 4.1 Docker
- `docker compose up -d && docker compose exec sscad bash` must give a shell in
  `/workdir` with everything ready. Nothing is built at image build time; the
  repository is mounted as a volume.
- Env vars set in compose **and** in `scripts/env.sh`:
  `TT_METAL_HOME=/tt-metal`, `TT_METAL_RUNTIME_ROOT=/tt-metal`,
  `ARCH_NAME=wormhole_b0`, `TT_METAL_SIMULATOR=<path in image>`,
  `TT_METAL_SLOW_DISPATCH_MODE=1`.
- **Check inside the image** (do not assume): that the tt-metal checkout is
  v0.77.0 (`git -C $TT_METAL_HOME describe --tags`), that the simulator `.so`
  exists, and which grid `compute_with_storage_grid_size()` reports under the
  simulator. Write the findings in the README "Ambiente" section.
- If the image is not v0.77.0, stop and report it; do not migrate the code to
  another version.

### 4.2 CMake
- Start from `petro-tt/tt-blas/blas/CMakeLists.txt` (include paths, link libs,
  CPM cache under `$TT_METAL_HOME/.cpmcache`), but trim it to what is needed:
  no OpenBLAS, no GTest in the skeleton.
- Add the tt-metal include directories as **`SYSTEM`** includes so that warnings
  from tt-metal headers do not show up, and **do not** pass
  `-Wno-deprecated-declarations`/`-Wno-deprecated`. Our own code must build
  with `-Wall -Wextra -Werror=deprecated-declarations`. This keeps the host code
  free of deprecated APIs.
- Each example's kernel directory is passed to the host code as a compile
  definition, e.g. `target_compile_definitions(ex02 PRIVATE KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels/")`,
  and `main.cpp` uses `KERNEL_DIR "dataflow/reader.cpp"` in `CreateKernel`.
  Check that `CreateKernel` accepts absolute paths in v0.77; if it doesn't,
  use the `OVERRIDE_KERNEL_PREFIX` approach from tt-blas and document it.
- Executables are named `ex00_dram_loopback`, `ex01_riscv_add`, `ex02_eltwise_add`, `ex03_…`, etc., and land in `build/bin/`.

### 4.3 Kernel deprecation check
Kernels are JIT-compiled at runtime, so host flags do not catch deprecated
kernel APIs. Add `scripts/check_kernels.sh`, which greps `examples/*/kernels`
for a forbidden-symbol list (`InterleavedAddrGenFast`, `InterleavedAddrGen<`,
`noc_async_read_tile`, `noc_async_write_tile`, `add_tiles_init`,
`mul_tiles_init`, `sub_tiles_init`) and fails if any is found. Note that the
upstream `add_2_integers_in_riscv` kernel uses `InterleavedAddrGen<true>`: do
not copy it as is (see §5.2). `run_all.sh` calls it first.

### 4.4 Listing anchors
The chapter includes code fragments via `\lstinputlisting`. Mark the important
regions with anchor comments that the LaTeX side will use:

```cpp
// ANCHOR: reader_loop
...
// ANCHOR_END: reader_loop
```

Anchor names: `snake_case`, unique per file. Minimum anchors per complete example:

| Ex. | Host (`main.cpp`) | Kernels |
|---|---|---|
| 00 | `host_buffers`, `host_kernel`, `host_runtime_args`, `host_enqueue` | `loopback_accessor`, `loopback_loop` |
| 01 | `host_buffers`, `host_kernel`, `host_runtime_args`, `host_enqueue` | `riscv_read`, `riscv_add`, `riscv_write` |
| 02 | `host_buffers`, `host_cbs`, `host_kernels`, `host_runtime_args`, `host_enqueue` | `reader_accessor`, `reader_loop`, `compute_loop`, `writer_loop` |

## 5. Complete examples

The three complete examples form a progression: **00** shows how data moves
(NoC, DRAM, L1, asynchronous transfers + barriers); **01** shows that the core
that moves data is also a programmable CPU, and why we still don't compute on
it; **02** introduces CBs and the compute kernel, where the real math happens.
Each README must point forward to the next example.

### 5.1 Example 00 — `00_dram_loopback`

**Goal:** first contact with TT-Metalium and with data movement. The smallest
program that puts data on the device, moves it with the NoC and brings it
back. Reference: `programming_examples/loopback` (already uses
`TensorAccessor`; follow it, translating the comments to pt-BR and expanding them).

- Operation: `out = in`, `in ∈ BF16^{num_tiles × 1024}`, default `num_tiles = 8`,
  configurable via argv. The data is opaque bytes here: no tiled layout
  conversion is needed (say so in a comment).
- **Single core** `{0,0}`, **one data movement kernel on `RISCV_0`**
  (`NOC::RISCV_0_default`), **no compute kernel and no CBs**.
- Host:
  1. Open a 1x1 `MeshDevice` (via `common/device.hpp`).
  2. Buffers: two interleaved DRAM `MeshBuffer`s (input, output), page = 1 BF16
     tile (2048 B), and **one L1 `MeshBuffer` of a single tile** used as scratch
     space by the kernel. Comment on why the L1 buffer is needed: a Tensix core
     cannot read DRAM directly; everything goes DRAM → (NoC) → L1 → (NoC) → DRAM.
     Comment on what "interleaved" means (pages distributed round-robin over the
     DRAM banks), since `TensorAccessor` hides this from the kernel.
  3. Compile-time args: `TensorAccessorArgs(*buf).append_to(args)` for input and output.
  4. Runtime args: L1 address, input address, output address, `num_tiles`.
  5. Fill the input with a fixed seed, `EnqueueWriteMeshBuffer` →
     `EnqueueMeshWorkload` → `EnqueueReadMeshBuffer` (blocking).
  6. Compare **bit-for-bit** with the input (a copy must be exact; no tolerance),
     print `PASSOU`/`FALHOU`, return non-zero on failure.
- Kernel `kernels/dataflow/loopback.cpp`:
  `TensorAccessorArgs<0>()` + `next_compile_time_args_offset()` for the output;
  per tile `noc_async_read_page(i, in, l1_addr) → noc_async_read_barrier() →
  noc_async_write_page(i, out, l1_addr) → noc_async_write_barrier()`.
  Comment on the asynchronous NoC model: the call only *issues* the transfer;
  the barrier is what guarantees completion, and without it the write could
  send a tile that has not arrived yet.
- The one-barrier-per-transfer loop is **deliberately serial** (a single L1
  slot, no overlap). Say so in a comment and in the README: example 02 fixes
  this with CBs and separate reader/writer kernels, and 06 revisits it for GEMM.
- Backs the chapter's introduction to the programming model / data movement
  (check the section number in `book_chapter_v2.tex`).

### 5.2 Example 01 — `01_riscv_add`

**Goal:** show that the data movement processors are ordinary (small) RISC-V
CPUs that run plain C++, and make their **limits** explicit, motivating the
compute kernel in example 02. Reference: `programming_examples/add_2_integers_in_riscv`,
**but that kernel uses the deprecated `InterleavedAddrGen<true>`**; rewrite the
DRAM access with `TensorAccessor`.

- Operation: `c = a + b`, `a, b, c ∈ uint32`, defaults `a = 14`, `b = 7`,
  configurable via argv.
- **Single core** `{0,0}`, **one data movement kernel on `RISCV_0`**, no compute
  kernel, no CBs.
- Host:
  1. Open a 1x1 `MeshDevice`.
  2. Buffers: three DRAM `MeshBuffer`s (`a`, `b`, `c`) and three L1 `MeshBuffer`s
     (scratch), each with `page_size = size = sizeof(uint32_t)`. Comment that the
     page here is one integer instead of a tile, and that the allocator still
     aligns each page to the DRAM/L1 alignment (verify the values in v0.77 and
     write them in the comment).
  3. Compile-time args: `TensorAccessorArgs` for `a`, `b`, `c`.
  4. Runtime args: DRAM addresses of `a`, `b`, `c` + L1 addresses of the three scratch buffers.
  5. Write `a` and `b`, enqueue, read `c` (blocking), compare with `a + b`
     computed on the host (uint32 wrap-around), print `PASSOU`/`FALHOU`.
  6. If `TT_METAL_DPRINT_CORES` is not set, print a hint (pt-BR) telling the user
     to `export TT_METAL_DPRINT_CORES=0,0` to see the kernel's message.
- Kernel `kernels/dataflow/riscv_add.cpp`:
  1. `riscv_read`: `noc_async_read_page(0, acc_a, l1_a)` and the same for `b`,
     then **one** `noc_async_read_barrier()` for both.
  2. `riscv_add`: read the operands through `volatile tt_l1_ptr uint32_t*`,
     `DPRINT` the operation, store `a + b` in `l1_c`. Then force the store to land
     in L1 before the NoC reads it (`ckernel::load_blocking(...)`, as in the
     upstream example; verify the header in v0.77). Comment why: the RISC-V
     and the NoC are independent L1 clients, and `fence` is a no-op on these
     cores (`tt-isa-documentation/WormholeB0/TensixTile/BabyRISCV/MemoryOrdering.md`).
  3. `riscv_write`: `noc_async_write_page(0, acc_c, l1_c)` → `noc_async_write_barrier()`.
- **Limitations to explain** (header comment of the kernel + a README section
  "Limitações do RISC-V"; take the facts from
  `tt-isa-documentation/WormholeB0/TensixTile/BabyRISCV/README.md`, not from memory):
  - each Tensix has 5 "baby" RISC-V cores: in-order, single-issue, **RV32IM**,
    ~1 GHz; two of them (B and NC) run the data movement kernels, the other
    three drive the compute engines;
  - **no floating point unit**: `float`/BF16 math would be emulated in software;
    only integer scalar work is cheap;
  - scalar, one element at a time: the throughput is orders of magnitude below
    the matrix (FPU) and vector (SFPU) engines used by compute kernels;
  - tiny local memory (2–4 KiB data RAM, small instruction cache) — the data lives in L1;
  - no direct DRAM access (everything via NoC into L1) and no automatic memory
    ordering between core stores and NoC transfers.
  Conclusion for the reader: use the RISC-Vs for control, addressing and
  bookkeeping (loop indices, address arithmetic, small integer work), and leave
  arithmetic on data to the compute kernel → example 02.

### 5.3 Example 02 — `02_eltwise_add`

**Goal:** the smallest complete program that shows the whole programming model:
host → DRAM buffers → reader → CB → compute → CB → writer → DRAM → host. It backs
chapter sections 3.1–3.4.

- Operation: `C = A + B`, `A, B, C ∈ BF16^{M×N}`, default `M = N = 64`
  (4 tiles), configurable via argv.
- **Single core** `{0,0}` (multicore is example 03).
- Host:
  1. Open a 1x1 `MeshDevice` (via `common/device.hpp`).
  2. Generate A and B with a fixed seed; convert to tiled layout (`common/layout.hpp`).
  3. Create 3 interleaved DRAM `MeshBuffer`s, page = 1 BF16 tile (2048 B).
  4. CBs: `c_0` (A), `c_1` (B), `c_16` (C), **2 tiles each (double buffering)**. Explain
     the choice of 2 in a comment.
  5. Reader on `RISCV_0`, writer on `RISCV_1`, compute with `ComputeConfig{}`.
     Compile-time args: `TensorAccessorArgs(*buf).append_to(args)` for each buffer
     (reader: A, B; writer: C), following `vecadd_multi_core.cpp`.
  6. Runtime args: buffer addresses + number of tiles.
  7. `EnqueueWriteMeshBuffer` → `EnqueueMeshWorkload` → `EnqueueReadMeshBuffer` (blocking).
  8. Convert back to row-major, compare with `reference.hpp` via `compare.hpp`
     (BF16 tolerance), print `PASSOU`/`FALHOU` and return non-zero on failure.
- Reader: `TensorAccessorArgs<0>()` / `next_compile_time_args_offset()`;
  loop `cb_reserve_back → get_write_ptr → noc_async_read_page(i, acc, addr)` for A and B →
  a single `noc_async_read_barrier()` for both reads → `cb_push_back`.
- Compute: `compute_kernel_hw_startup(c_0, c_1, c_16)`, `add_init(c_0, c_1)`
  (check the exact signature in v0.77),
  `tile_regs_acquire → add_tiles → tile_regs_commit → tile_regs_wait → pack_tile → tile_regs_release`,
  with `cb_wait_front/cb_pop_front/cb_reserve_back/cb_push_back`. Add a comment on
  the Dst register handshake (MATH ↔ PACK), since the chapter explains this.
- Writer: `cb_wait_front → get_read_ptr → noc_async_write_page → noc_async_writes_flushed`
  per tile, `noc_async_write_barrier()` once at the end, `cb_pop_front`.
- Optional: one `DPRINT` line guarded by a macro to demonstrate debugging (§3.6
  of the chapter); off by default.

### 5.4 Example 03 — `03_eltwise_multicore`

Example 02 split across N cores with `split_work_to_cores` (2 core groups),
following `programming_examples/vecadd_multi_core` and the work-group loop of
`petro-tt/tt-blas` (`axpy.cpp`). CLI: `[M] [N] [max_cores]`, default
`256x256`. Reader and writer take `start_tile` + `num_tiles`; the compute
kernel is identical to example 02. Anchors: `host_split`,
`host_runtime_args`, `reader_loop`, `compute_loop`, `writer_loop`.

## 6. Stubs: contents of each README

Every stub README must already say (in pt-BR), so later tasks are unambiguous:

| Ex. | Content to implement later | Reference |
|---|---|---|
| 04 | GEMV `y = αAx + βy`, BF16 in / FP32 acc, multicore over rows of tiles | `petro-tt/tt-blas/blas/gemv` |
| 05 | GEMM v1: faithful port of the tt-blas GEMM (1 output tile per iteration, barrier per tile read, no reuse), with `TensorAccessor` | `petro-tt/tt-blas/blas/gemm` |
| 06 | GEMM v2: v1 with CBs ≥ 2 tiles and reads of A/B batched under one barrier (latency overlap) | — |
| 07 | GEMM v3: output blocks per core, A/B blocks kept in L1 for reuse (`in0_block_w`, `out_subblock_h/w`) | `programming_examples/matmul/matmul_multicore_reuse` |
| 08 | GEMM v4: v3 + multicast of A along rows and B along columns, with semaphores | `programming_examples/matmul/matmul_multicore_reuse_mcast` |
| 09 | `Y = GELU(XW + b)`: GEMM v3 + bias add + GELU in SFPU, fused into the compute kernel | `api/compute/eltwise_unary/gelu.h` |

All GEMM examples must share the same CLI (`--m --n --k --iters --check`) and print
one CSV line `example,m,n,k,dtype,cores,time_ms,gflops,max_rel_err`, so the
benchmark can collect them later.

## 7. Acceptance criteria

1. From a clean clone: `docker compose up -d`, `docker compose exec sscad bash`,
   `./scripts/build.sh` and `./scripts/run_all.sh` complete with exit 0.
2. On the simulator: `ex00_dram_loopback` prints `PASSOU` for the default size
   and for `num_tiles = 64`; `ex01_riscv_add` prints `PASSOU` for the defaults
   and for `a = 0xFFFFFFFF, b = 1` (wrap-around), and with
   `TT_METAL_DPRINT_CORES=0,0` the kernel's `DPRINT` line appears (if DPRINT does
   not work under ttsim, record it in "Problemas comuns"); `ex02_eltwise_add`
   prints `PASSOU` for the default size and for `M=N=128`; `ex03_eltwise_multicore`
   prints `PASSOU` for the default size and for an uneven split (e.g. `512x512`).
3. No use of `InterleavedAddrGenFast`/`InterleavedAddrGen` or `*_tiles_init`
   anywhere (`check_kernels.sh` green); host builds with
   `-Werror=deprecated-declarations`.
4. The README (pt-BR) has: quick start, list of examples with status
   (implemented/stub), the "Ambiente" section (image + digest, tt-metal
   version, simulator grid), and a "Problemas comuns" section listing the
   pitfalls actually encountered.
5. The anchors from §4.4 exist in examples 00, 01 and 02 (and §5.4 in 03).
6. The README of example 01 has the "Limitações do RISC-V" section (§5.2).
7. Commits are small and descriptive, on a branch `feat/skeleton`. **Do not push
   and do not open a PR** without the owner's approval.

## 8. Out of scope
- Implementing examples 04–09.
- Benchmarks on real hardware and plots.
- Any change to `petro-tt` or to the chapter.
- Supporting Blackhole or real hardware (it may work, but it is not tested here).

## 9. Final report from the executing agent
At the end, report: what was done, the exact commands that were validated, the
tt-metal version and simulator grid found, deviations from this spec (with the
reason), and open problems.
