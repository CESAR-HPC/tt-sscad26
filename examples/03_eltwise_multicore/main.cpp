// SPDX-License-Identifier: Apache-2.0
//
// Exemplo 03 — soma elemento a elemento C = A + B em vários núcleos.
//
// Mesmo programa do exemplo 02, mas os tiles são divididos entre os núcleos
// Tensix do chip. Cada núcleo executa os mesmos três kernels sobre uma faixa
// contígua de tiles [start_tile, start_tile + num_tiles). As únicas mudanças
// em relação ao exemplo 02 são a divisão do trabalho no host e o índice
// inicial passado a cada kernel.

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

using namespace tt::tt_metal;

int main() {
    // 512x512 = 256 tiles, que não se dividem igualmente entre os núcleos:
    // assim o exemplo mostra os dois grupos de split_work_to_cores.
    constexpr uint32_t m = 512;
    constexpr uint32_t n = 512;
    static_assert(m % tt::constants::TILE_HEIGHT == 0 && n % tt::constants::TILE_WIDTH == 0);
    constexpr uint32_t num_tiles = (m / tt::constants::TILE_HEIGHT) * (n / tt::constants::TILE_WIDTH);
    constexpr uint32_t tile_bytes = sizeof(bfloat16) * tt::constants::TILE_HW;

    std::vector<bfloat16> a(m * n);
    std::vector<bfloat16> b(m * n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (bfloat16 &value : a) {
        value = bfloat16(dist(rng));
    }
    for (bfloat16 &value : b) {
        value = bfloat16(dist(rng));
    }

    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue &cq = mesh_device->mesh_command_queue();

    // ANCHOR: host_split
    // Grade de núcleos Tensix disponíveis para computação.
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();

    // split_work_to_cores divide num_tiles entre os núcleos da grade. Quando
    // a divisão não é exata, há dois grupos: group_1 recebe tiles_per_core_1
    // tiles por núcleo e group_2 recebe um a menos (tiles_per_core_2).
    uint32_t num_cores = 0;
    tt::tt_metal::CoreRangeSet all_cores;
    tt::tt_metal::CoreRangeSet group_1;
    tt::tt_metal::CoreRangeSet group_2;
    uint32_t tiles_per_core_1 = 0;
    uint32_t tiles_per_core_2 = 0;
    std::tie(num_cores, all_cores, group_1, group_2, tiles_per_core_1, tiles_per_core_2) =
        split_work_to_cores(grid, num_tiles);
    // ANCHOR_END: host_split

    fmt::print("Exemplo 03: C = A + B com {}x{} bfloat16 ({} tiles) em {} de {}x{} núcleos\n", m,
               n, num_tiles, num_cores, grid.x, grid.y);
    fmt::print("  {} núcleos com {} tiles + {} núcleos com {} tiles\n", group_1.num_cores(),
               tiles_per_core_1, group_2.num_cores(), tiles_per_core_2);

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = tile_bytes, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig size{.size = num_tiles * tile_bytes};

    std::shared_ptr<distributed::MeshBuffer> a_buf =
        distributed::MeshBuffer::create(size, dram_config, mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> b_buf =
        distributed::MeshBuffer::create(size, dram_config, mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> c_buf =
        distributed::MeshBuffer::create(size, dram_config, mesh_device.get());

    Program program = CreateProgram();

    // Os CBs e kernels agora são criados em todos os núcleos de uma vez,
    // passando o CoreRangeSet em vez de um único CoreCoord.
    constexpr uint32_t tiles_per_cb = 2; // double buffering, como no exemplo 02

    CircularBufferConfig cb_a_config =
        CircularBufferConfig(tiles_per_cb * tile_bytes, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_0, tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_a_config);

    CircularBufferConfig cb_b_config =
        CircularBufferConfig(tiles_per_cb * tile_bytes, {{tt::CBIndex::c_1, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_1, tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_b_config);

    CircularBufferConfig cb_c_config =
        CircularBufferConfig(tiles_per_cb * tile_bytes, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_16, tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_c_config);

    std::vector<uint32_t> reader_args;
    TensorAccessorArgs(*a_buf).append_to(reader_args);
    TensorAccessorArgs(*b_buf).append_to(reader_args);
    std::vector<uint32_t> writer_args;
    TensorAccessorArgs(*c_buf).append_to(writer_args);

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
    KernelHandle compute =
        CreateKernel(program, KERNEL_DIR "compute/compute.cpp", all_cores, ComputeConfig{});

    // ANCHOR: host_runtime_args
    // Cada núcleo recebe a sua faixa de tiles. Percorremos os dois grupos na
    // mesma ordem usada por split_work_to_cores, acumulando o tile inicial.
    const std::vector<std::pair<tt::tt_metal::CoreRangeSet, uint32_t>> work_groups = {
        {group_1, tiles_per_core_1}, {group_2, tiles_per_core_2}};

    uint32_t start_tile = 0;
    for (const std::pair<tt::tt_metal::CoreRangeSet, uint32_t> &work_group : work_groups) {
        const uint32_t tiles_per_core = work_group.second;
        for (const tt::tt_metal::CoreRange &range : work_group.first.ranges()) {
            for (const CoreCoord &core : range) {
                SetRuntimeArgs(program, reader, core,
                               {static_cast<uint32_t>(a_buf->address()),
                                static_cast<uint32_t>(b_buf->address()), start_tile,
                                tiles_per_core});
                SetRuntimeArgs(program, compute, core, {tiles_per_core});
                SetRuntimeArgs(program, writer, core,
                               {static_cast<uint32_t>(c_buf->address()), start_tile,
                                tiles_per_core});
                start_tile += tiles_per_core;
            }
        }
    }
    // ANCHOR_END: host_runtime_args

    const std::vector<bfloat16> a_tiled = convert_layout<bfloat16>(
        a, {m, n}, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES);
    const std::vector<bfloat16> b_tiled = convert_layout<bfloat16>(
        b, {m, n}, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES);
    distributed::EnqueueWriteMeshBuffer(cq, a_buf, a_tiled, /*blocking=*/false);
    distributed::EnqueueWriteMeshBuffer(cq, b_buf, b_tiled, /*blocking=*/false);

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()),
                         std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

    std::vector<bfloat16> c_tiled;
    distributed::EnqueueReadMeshBuffer(cq, c_tiled, c_buf, /*blocking=*/true);

    const std::vector<bfloat16> c = convert_layout<bfloat16>(
        c_tiled, {m, n}, TensorLayoutType::TILED_NFACES, TensorLayoutType::LIN_ROW_MAJOR);
    const sscad::CompareResult result = sscad::compare(c, sscad::reference::eltwise_add(a, b));
    sscad::print_compare(result);

    const bool closed = mesh_device->close();
    return sscad::report(closed && result.pass);
}
