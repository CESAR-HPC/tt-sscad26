// Exemplo 05 — GEMM v1 (ingênuo): C = A B.
//
// A e B são bfloat16, C é float e a acumulação é feita em FP32 no Dst. Os
// tiles de C são divididos entre os núcleos; para cada tile de saída (i, j) o
// núcleo lê os Kt tiles da linha i de A e da coluna j de B, um de cada vez,
// com uma barreira por leitura, e nada é reaproveitado entre tiles de saída.
// É a versão mais simples possível, e o ponto de partida dos exemplos 06–08.

#include <cstdint>
#include <memory>
#include <random>
#include <tuple>
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
#include <tt-metalium/work_split.hpp>

#include "compare.hpp"
#include "reference.hpp"
#include "timer.hpp"

using namespace tt::tt_metal;

int main() {
    // A é m x k, B é k x n, C é m x n. Os três tamanhos são diferentes para
    // que os índices não se confundam. São os mesmos tamanhos dos exemplos
    // 06–09: com os blocos de 2x2 tiles dos exemplos 07–09, os 18x16 tiles de
    // C formam uma grade de 8x9 blocos, que ocupa os 72 núcleos. Aqui os 288
    // tiles se dividem igualmente (4 por núcleo, o grupo 2 fica vazio).
    constexpr uint32_t m = 576;
    constexpr uint32_t n = 512;
    constexpr uint32_t k = 384;
    static_assert(m % tt::constants::TILE_HEIGHT == 0 &&
                  n % tt::constants::TILE_WIDTH == 0 &&
                  k % tt::constants::TILE_WIDTH == 0);
    constexpr uint32_t Mt = m / tt::constants::TILE_HEIGHT;
    constexpr uint32_t Nt = n / tt::constants::TILE_WIDTH;
    constexpr uint32_t Kt = k / tt::constants::TILE_WIDTH;
    constexpr uint32_t num_output_tiles = Mt * Nt;

    constexpr uint32_t in_tile_bytes =
        sizeof(bfloat16) * tt::constants::TILE_HW;
    constexpr uint32_t out_tile_bytes = sizeof(float) * tt::constants::TILE_HW;

    std::vector<bfloat16> a(m * k);
    std::vector<bfloat16> b(k * n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (bfloat16 &value : a) {
        value = bfloat16(dist(rng));
    }
    for (bfloat16 &value : b) {
        value = bfloat16(dist(rng));
    }

    std::shared_ptr<distributed::MeshDevice> mesh_device =
        distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue &cq = mesh_device->mesh_command_queue();

    // ANCHOR: host_split
    // A unidade de trabalho é um tile de C. Cada núcleo recebe uma faixa
    // contígua [start_tile, start_tile + num_tiles) de tiles de C, numerados
    // em row-major (tile t = linha t / Nt, coluna t % Nt).
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();

    uint32_t num_cores = 0;
    tt::tt_metal::CoreRangeSet all_cores;
    tt::tt_metal::CoreRangeSet group_1;
    tt::tt_metal::CoreRangeSet group_2;
    uint32_t tiles_per_core_1 = 0;
    uint32_t tiles_per_core_2 = 0;
    std::tie(num_cores, all_cores, group_1, group_2, tiles_per_core_1,
             tiles_per_core_2) = split_work_to_cores(grid, num_output_tiles);
    // ANCHOR_END: host_split

    fmt::print("Exemplo 05: C = A B com A {}x{}, B {}x{} bfloat16 e C float em "
               "{} de {}x{} núcleos\n",
               m, k, k, n, num_cores, grid.x, grid.y);
    fmt::print("  {} núcleos com {} tiles + {} núcleos com {} tiles\n",
               group_1.num_cores(), tiles_per_core_1, group_2.num_cores(),
               tiles_per_core_2);

    // ANCHOR: host_buffers
    distributed::DeviceLocalBufferConfig in_dram_config{
        .page_size = in_tile_bytes, .buffer_type = BufferType::DRAM};
    distributed::DeviceLocalBufferConfig out_dram_config{
        .page_size = out_tile_bytes, .buffer_type = BufferType::DRAM};

    distributed::ReplicatedBufferConfig a_size{.size = Mt * Kt * in_tile_bytes};
    distributed::ReplicatedBufferConfig b_size{.size = Kt * Nt * in_tile_bytes};
    distributed::ReplicatedBufferConfig c_size{.size =
                                                   Mt * Nt * out_tile_bytes};

    std::shared_ptr<distributed::MeshBuffer> a_buf =
        distributed::MeshBuffer::create(a_size, in_dram_config,
                                        mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> b_buf =
        distributed::MeshBuffer::create(b_size, in_dram_config,
                                        mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> c_buf =
        distributed::MeshBuffer::create(c_size, out_dram_config,
                                        mesh_device.get());
    // ANCHOR_END: host_buffers

    Program program = CreateProgram();

    // ANCHOR: host_cbs
    // CBs de 1 tile: o reader só lê o próximo tile depois que o compute
    // consumiu o anterior, e não há sobreposição entre leitura e cálculo. O
    // exemplo 06 aumenta os CBs para permitir essa sobreposição.
    constexpr uint32_t tiles_per_cb = 1;

    CircularBufferConfig cb_a_config =
        CircularBufferConfig(tiles_per_cb * in_tile_bytes,
                             {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_0, in_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_a_config);

    CircularBufferConfig cb_b_config =
        CircularBufferConfig(tiles_per_cb * in_tile_bytes,
                             {{tt::CBIndex::c_1, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_1, in_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_b_config);

    CircularBufferConfig cb_c_config =
        CircularBufferConfig(tiles_per_cb * out_tile_bytes,
                             {{tt::CBIndex::c_16, tt::DataFormat::Float32}})
            .set_page_size(tt::CBIndex::c_16, out_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_c_config);
    // ANCHOR_END: host_cbs

    std::vector<uint32_t> reader_args;
    TensorAccessorArgs(*a_buf).append_to(reader_args);
    TensorAccessorArgs(*b_buf).append_to(reader_args);
    std::vector<uint32_t> writer_args;
    TensorAccessorArgs(*c_buf).append_to(writer_args);

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
    // fp32_dest_acc_en: a soma dos Kt produtos de tiles é feita em FP32.
    KernelHandle compute = CreateKernel(
        program, KERNEL_DIR "compute/compute.cpp", all_cores,
        ComputeConfig{.math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
                      .fp32_dest_acc_en = true});
    // ANCHOR_END: host_kernels

    // ANCHOR: host_runtime_args
    const std::vector<std::pair<tt::tt_metal::CoreRangeSet, uint32_t>>
        work_groups = {{group_1, tiles_per_core_1},
                       {group_2, tiles_per_core_2}};

    uint32_t start_tile = 0;
    for (const std::pair<tt::tt_metal::CoreRangeSet, uint32_t> &work_group :
         work_groups) {
        const uint32_t tiles_per_core = work_group.second;
        for (const tt::tt_metal::CoreRange &range : work_group.first.ranges()) {
            for (const CoreCoord &core : range) {
                SetRuntimeArgs(program, reader, core,
                               {static_cast<uint32_t>(a_buf->address()),
                                static_cast<uint32_t>(b_buf->address()), Kt, Nt,
                                start_tile, tiles_per_core});
                SetRuntimeArgs(program, compute, core, {Kt, tiles_per_core});
                SetRuntimeArgs(program, writer, core,
                               {static_cast<uint32_t>(c_buf->address()),
                                start_tile, tiles_per_core});
                start_tile += tiles_per_core;
            }
        }
    }
    // ANCHOR_END: host_runtime_args

    const std::vector<bfloat16> a_tiled =
        convert_layout<bfloat16>(a, {m, k}, TensorLayoutType::LIN_ROW_MAJOR,
                                 TensorLayoutType::TILED_NFACES);
    const std::vector<bfloat16> b_tiled =
        convert_layout<bfloat16>(b, {k, n}, TensorLayoutType::LIN_ROW_MAJOR,
                                 TensorLayoutType::TILED_NFACES);

    // ANCHOR: host_enqueue
    distributed::EnqueueWriteMeshBuffer(cq, a_buf, a_tiled, /*blocking=*/false);
    distributed::EnqueueWriteMeshBuffer(cq, b_buf, b_tiled, /*blocking=*/false);

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()),
                         std::move(program));

    // A primeira execução inclui a compilação JIT dos kernels; medimos a segunda.
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
    sscad::Timer timer;
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
    const double time_ms = timer.elapsed_ms();

    std::vector<float> c_tiled;
    distributed::EnqueueReadMeshBuffer(cq, c_tiled, c_buf, /*blocking=*/true);
    // ANCHOR_END: host_enqueue

    const std::vector<float> c =
        convert_layout<float>(c_tiled, {m, n}, TensorLayoutType::TILED_NFACES,
                              TensorLayoutType::LIN_ROW_MAJOR);
    const sscad::CompareResult result =
        sscad::compare(c, sscad::reference::gemm(m, n, k, a, b));
    sscad::print_compare(result);

    // Linha CSV para o benchmark: 2 m n k operações de ponto flutuante.
    const double gflops = 2.0 * m * n * k / (time_ms * 1e6);
    fmt::print("example,m,n,k,dtype,cores,time_ms,gflops,max_rel_err\n");
    fmt::print("05_gemm_v1_naive,{},{},{},bf16-fp32,{},{:.3f},{:.3f},{:.3e}\n",
               m, n, k, num_cores, time_ms, gflops, result.max_rel_err);

    const bool closed = mesh_device->close();
    return sscad::report(closed && result.pass);
}
