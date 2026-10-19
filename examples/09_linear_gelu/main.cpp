// SPDX-License-Identifier: Apache-2.0
//
// Exemplo 09 — Camada linear + GELU: Y = GELU(X W + b).
//
// O GEMM é o do exemplo 07 (blocos de X e W reaproveitados em L1, somas
// parciais em c_24), com X no lugar de A, W no lugar de B e Y no lugar de C.
// A soma do bias e a GELU são fundidas no kernel de computação: o resultado
// de X W não volta para a DRAM. Depois do último passo em K, cada sub-bloco
// de X W é somado ao bias (na FPU, com broadcast da linha do bias para as 32
// linhas do tile) e passa pela GELU (na SFPU) ainda no Dst, antes de ser
// empacotado para o writer.
//
// X, W e b são bfloat16; Y é float, e a acumulação é feita em FP32 no Dst.

#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include <fmt/core.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>

#include "compare.hpp"
#include "reference.hpp"
#include "timer.hpp"

using namespace tt::tt_metal;

int main() {
    // X é m x k (m entradas de dimensão k), W é k x n, b tem n elementos e Y
    // é m x n. Os mesmos tamanhos dos exemplos 05–08: com os blocos de 2x2
    // tiles, a grade de 8x9 blocos ocupa os 72 núcleos do simulador.
    constexpr uint32_t m = 576;
    constexpr uint32_t n = 512;
    constexpr uint32_t k = 384;
    static_assert(m % tt::constants::TILE_HEIGHT == 0 &&
                  n % tt::constants::TILE_WIDTH == 0 &&
                  k % tt::constants::TILE_WIDTH == 0);
    constexpr uint32_t Mt = m / tt::constants::TILE_HEIGHT;
    constexpr uint32_t Nt = n / tt::constants::TILE_WIDTH;
    constexpr uint32_t Kt = k / tt::constants::TILE_WIDTH;

    // Os blocos do exemplo 07 (ver lá a explicação de cada parâmetro).
    constexpr uint32_t per_core_M = 2;
    constexpr uint32_t per_core_N = 2;
    constexpr uint32_t in0_block_w = 2;
    constexpr uint32_t out_subblock_h = 1;
    constexpr uint32_t out_subblock_w = 2;
    static_assert(Mt % per_core_M == 0 && Nt % per_core_N == 0 &&
                  Kt % in0_block_w == 0);
    static_assert(per_core_M % out_subblock_h == 0 &&
                  per_core_N % out_subblock_w == 0);
    static_assert(out_subblock_h * out_subblock_w <= 4);

    constexpr uint32_t num_k_blocks = Kt / in0_block_w;
    constexpr uint32_t num_blocks_y = Mt / per_core_M;
    constexpr uint32_t num_blocks_x = Nt / per_core_N;

    constexpr uint32_t in_tile_bytes =
        sizeof(bfloat16) * tt::constants::TILE_HW;
    constexpr uint32_t out_tile_bytes = sizeof(float) * tt::constants::TILE_HW;

    std::vector<bfloat16> x(m * k);
    std::vector<bfloat16> w(k * n);
    std::vector<bfloat16> bias(n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (bfloat16 &value : x) {
        value = bfloat16(dist(rng));
    }
    for (bfloat16 &value : w) {
        value = bfloat16(dist(rng));
    }
    for (bfloat16 &value : bias) {
        value = bfloat16(dist(rng));
    }

    // ANCHOR: host_bias_layout
    // O bias é um vetor, mas a FPU só trabalha com tiles de 32 x 32. Ele vira
    // uma matriz 32 x n em que só a linha 0 tem dados (o resto é zero), ou
    // seja, Nt tiles com o bias na primeira linha. No compute,
    // add_tiles_bcast_rows repete essa linha nas 32 linhas do tile de X W.
    std::vector<bfloat16> bias_rows(tt::constants::TILE_HEIGHT * n,
                                    bfloat16(0.0f));
    for (uint32_t j = 0; j < n; j++) {
        bias_rows[j] = bias[j];
    }
    // ANCHOR_END: host_bias_layout

    std::shared_ptr<distributed::MeshDevice> mesh_device =
        distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue &cq = mesh_device->mesh_command_queue();

    // Um bloco de Y por núcleo: o bloco (by, bx) fica no núcleo lógico
    // (x = bx, y = by), como no exemplo 07.
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    if (num_blocks_x > grid.x || num_blocks_y > grid.y) {
        fmt::print("A grade de blocos {}x{} não cabe na grade de núcleos {}x{}\n",
                   num_blocks_x, num_blocks_y, grid.x, grid.y);
        return 1;
    }
    const tt::tt_metal::CoreRange all_cores({0, 0},
                                            {num_blocks_x - 1, num_blocks_y - 1});
    constexpr uint32_t num_cores = num_blocks_x * num_blocks_y;

    fmt::print("Exemplo 09: Y = GELU(X W + b) com X {}x{}, W {}x{} e b {} "
               "bfloat16 e Y float em {} de {}x{} núcleos\n",
               m, k, k, n, n, num_cores, grid.x, grid.y);

    // ANCHOR: host_buffers
    distributed::DeviceLocalBufferConfig in_dram_config{
        .page_size = in_tile_bytes, .buffer_type = BufferType::DRAM};
    distributed::DeviceLocalBufferConfig out_dram_config{
        .page_size = out_tile_bytes, .buffer_type = BufferType::DRAM};

    distributed::ReplicatedBufferConfig x_size{.size = Mt * Kt * in_tile_bytes};
    distributed::ReplicatedBufferConfig w_size{.size = Kt * Nt * in_tile_bytes};
    distributed::ReplicatedBufferConfig bias_size{.size = Nt * in_tile_bytes};
    distributed::ReplicatedBufferConfig y_size{.size =
                                                   Mt * Nt * out_tile_bytes};

    std::shared_ptr<distributed::MeshBuffer> x_buf =
        distributed::MeshBuffer::create(x_size, in_dram_config,
                                        mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> w_buf =
        distributed::MeshBuffer::create(w_size, in_dram_config,
                                        mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> bias_buf =
        distributed::MeshBuffer::create(bias_size, in_dram_config,
                                        mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> y_buf =
        distributed::MeshBuffer::create(y_size, out_dram_config,
                                        mesh_device.get());
    // ANCHOR_END: host_buffers

    Program program = CreateProgram();

    // ANCHOR: host_cbs
    constexpr uint32_t x_block_tiles = per_core_M * in0_block_w;
    constexpr uint32_t w_block_tiles = in0_block_w * per_core_N;
    constexpr uint32_t out_block_tiles = per_core_M * per_core_N;

    // X e W: dois blocos cada (double buffering), como no exemplo 07.
    CircularBufferConfig cb_x_config =
        CircularBufferConfig(2 * x_block_tiles * in_tile_bytes,
                             {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_0, in_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_x_config);

    CircularBufferConfig cb_w_config =
        CircularBufferConfig(2 * w_block_tiles * in_tile_bytes,
                             {{tt::CBIndex::c_1, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_1, in_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_w_config);

    // Bias: os per_core_N tiles das colunas do bloco. São lidos uma vez e
    // usados por todas as linhas do bloco.
    CircularBufferConfig cb_bias_config =
        CircularBufferConfig(per_core_N * in_tile_bytes,
                             {{tt::CBIndex::c_2, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_2, in_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_bias_config);

    // Y: o bloco de saída inteiro.
    CircularBufferConfig cb_y_config =
        CircularBufferConfig(out_block_tiles * out_tile_bytes,
                             {{tt::CBIndex::c_16, tt::DataFormat::Float32}})
            .set_page_size(tt::CBIndex::c_16, out_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_y_config);

    // Somas parciais de X W em float. Depois do último passo em K, guardam o
    // X W completo, que a FPU lê de volta para somar o bias: a soma com
    // broadcast lê os dois operandos de CBs, não do Dst.
    CircularBufferConfig cb_partial_config =
        CircularBufferConfig(out_block_tiles * out_tile_bytes,
                             {{tt::CBIndex::c_24, tt::DataFormat::Float32}})
            .set_page_size(tt::CBIndex::c_24, out_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_partial_config);
    // ANCHOR_END: host_cbs

    std::vector<uint32_t> reader_args;
    TensorAccessorArgs(*x_buf).append_to(reader_args);
    TensorAccessorArgs(*w_buf).append_to(reader_args);
    TensorAccessorArgs(*bias_buf).append_to(reader_args);
    std::vector<uint32_t> writer_args;
    TensorAccessorArgs(*y_buf).append_to(writer_args);
    const std::vector<uint32_t> compute_args = {
        in0_block_w,    num_k_blocks,   per_core_M,
        per_core_N,     out_subblock_h, out_subblock_w};

    // ANCHOR: host_kernels
    KernelHandle reader = CreateKernel(
        program, KERNEL_DIR "dataflow/reader.cpp", all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc = NOC::RISCV_0_default,
                           .compile_args = reader_args});
    KernelHandle writer = CreateKernel(
        program, KERNEL_DIR "dataflow/writer.cpp", all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                           .noc = NOC::RISCV_1_default,
                           .compile_args = writer_args});
    CreateKernel(program, KERNEL_DIR "compute/compute.cpp", all_cores,
                 ComputeConfig{.math_fidelity =
                                   tt::tt_metal::MathFidelity::HiFi4,
                               .fp32_dest_acc_en = true,
                               .compile_args = compute_args});
    // ANCHOR_END: host_kernels

    // ANCHOR: host_runtime_args
    for (uint32_t by = 0; by < num_blocks_y; by++) {
        for (uint32_t bx = 0; bx < num_blocks_x; bx++) {
            const CoreCoord core = {bx, by};
            SetRuntimeArgs(program, reader, core,
                           {static_cast<uint32_t>(x_buf->address()),
                            static_cast<uint32_t>(w_buf->address()),
                            static_cast<uint32_t>(bias_buf->address()), Kt,
                            Nt, by, bx, per_core_M, per_core_N, in0_block_w});
            SetRuntimeArgs(program, writer, core,
                           {static_cast<uint32_t>(y_buf->address()), Nt, by,
                            bx, per_core_M, per_core_N, out_subblock_h,
                            out_subblock_w});
        }
    }
    // ANCHOR_END: host_runtime_args

    const std::vector<bfloat16> x_tiled =
        convert_layout<bfloat16>(x, {m, k}, TensorLayoutType::LIN_ROW_MAJOR,
                                 TensorLayoutType::TILED_NFACES);
    const std::vector<bfloat16> w_tiled =
        convert_layout<bfloat16>(w, {k, n}, TensorLayoutType::LIN_ROW_MAJOR,
                                 TensorLayoutType::TILED_NFACES);
    const std::vector<bfloat16> bias_tiled = convert_layout<bfloat16>(
        bias_rows, {tt::constants::TILE_HEIGHT, n},
        TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES);

    // ANCHOR: host_enqueue
    distributed::EnqueueWriteMeshBuffer(cq, x_buf, x_tiled, /*blocking=*/false);
    distributed::EnqueueWriteMeshBuffer(cq, w_buf, w_tiled, /*blocking=*/false);
    distributed::EnqueueWriteMeshBuffer(cq, bias_buf, bias_tiled,
                                        /*blocking=*/false);

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()),
                         std::move(program));

    // A primeira execução inclui a compilação JIT dos kernels; medimos a segunda.
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
    sscad::Timer timer;
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
    const double time_ms = timer.elapsed_ms();

    std::vector<float> y_tiled;
    distributed::EnqueueReadMeshBuffer(cq, y_tiled, y_buf, /*blocking=*/true);
    // ANCHOR_END: host_enqueue

    const std::vector<float> y =
        convert_layout<float>(y_tiled, {m, n}, TensorLayoutType::TILED_NFACES,
                              TensorLayoutType::LIN_ROW_MAJOR);
    const sscad::CompareResult result =
        sscad::compare(y, sscad::reference::linear_gelu(m, n, k, x, w, bias));
    sscad::print_compare(result);

    // Linha CSV para o benchmark. Contamos só as 2 m n k operações do GEMM;
    // o bias e a GELU custam O(m n).
    const double gflops = 2.0 * m * n * k / (time_ms * 1e6);
    fmt::print("example,m,n,k,dtype,cores,time_ms,gflops,max_rel_err\n");
    fmt::print("09_linear_gelu,{},{},{},bf16-fp32,{},{:.3f},{:.3f},{:.3e}\n",
               m, n, k, num_cores, time_ms, gflops, result.max_rel_err);

    const bool closed = mesh_device->close();
    return sscad::report(closed && result.pass);
}
