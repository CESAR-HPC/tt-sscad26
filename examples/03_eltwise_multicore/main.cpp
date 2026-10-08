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
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <fmt/core.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "compare.hpp"
#include "data_format.hpp"
#include "device.hpp"
#include "layout.hpp"
#include "reference.hpp"

using namespace tt::tt_metal;

int main(int argc, char **argv) {
    // Uso: ex03_eltwise_multicore [M] [N] [max_cores]
    const uint32_t m = argc > 1 ? std::stoul(argv[1]) : 256;
    const uint32_t n = argc > 2 ? std::stoul(argv[2]) : 256;
    const uint32_t max_cores = argc > 3 ? std::stoul(argv[3]) : 0; // 0 = todos

    const uint32_t mp = sscad::round_up_to_tile(m);
    const uint32_t np = sscad::round_up_to_tile(n);
    const uint32_t num_tiles = (mp / sscad::TILE) * (np / sscad::TILE);
    constexpr uint32_t tile_bytes = sscad::tile_size_bytes<bfloat16>();

    std::vector<bfloat16> a(m * n), b(m * n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : a) v = bfloat16(dist(rng));
    for (auto &v : b) v = bfloat16(dist(rng));

    auto mesh_device = sscad::open_device();
    distributed::MeshCommandQueue &cq = mesh_device->mesh_command_queue();

    // ANCHOR: host_split
    // split_work_to_cores divide num_tiles entre os núcleos da grade. Quando
    // a divisão não é exata, há dois grupos: group_1 recebe tiles_per_core_1
    // tiles por núcleo e group_2 recebe um a menos (tiles_per_core_2).
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    const uint32_t grid_cores = grid.x * grid.y;
    const uint32_t use_cores = (max_cores > 0 && max_cores < grid_cores) ? max_cores : grid_cores;
    const tt::tt_metal::CoreRangeSet available = num_cores_to_corerangeset(use_cores, grid, /*row_wise=*/true);

    auto [num_cores, all_cores, group_1, group_2, tiles_per_core_1, tiles_per_core_2] =
        split_work_to_cores(available, num_tiles, /*row_wise=*/true);
    // ANCHOR_END: host_split

    fmt::print("Exemplo 03: C = A + B com {}x{} bfloat16 ({} tiles) em {} núcleos\n", m, n,
               num_tiles, num_cores);
    fmt::print("  {} núcleos com {} tiles + {} núcleos com {} tiles\n", group_1.num_cores(),
               tiles_per_core_1, group_2.num_cores(), tiles_per_core_2);

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = tile_bytes, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig size{.size = num_tiles * tile_bytes};

    auto a_buf = distributed::MeshBuffer::create(size, dram_config, mesh_device.get());
    auto b_buf = distributed::MeshBuffer::create(size, dram_config, mesh_device.get());
    auto c_buf = distributed::MeshBuffer::create(size, dram_config, mesh_device.get());

    Program program = CreateProgram();

    // Os CBs e kernels agora são criados em todos os núcleos de uma vez,
    // passando o CoreRangeSet em vez de um único CoreCoord.
    constexpr uint32_t tiles_per_cb = 2; // double buffering, como no exemplo 02
    auto make_cb = [&](tt::CBIndex index) {
        CircularBufferConfig config =
            CircularBufferConfig(tiles_per_cb * tile_bytes,
                                 {{index, sscad::data_format_of<bfloat16>()}})
                .set_page_size(index, tile_bytes);
        CreateCircularBuffer(program, all_cores, config);
    };
    make_cb(tt::CBIndex::c_0);
    make_cb(tt::CBIndex::c_1);
    make_cb(tt::CBIndex::c_16);

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
    for (const auto &[group, tiles_per_core] : work_groups) {
        for (const tt::tt_metal::CoreRange &range : group.ranges()) {
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

    distributed::EnqueueWriteMeshBuffer(cq, a_buf, sscad::tilize(sscad::pad_to_tiles(a, m, n), mp, np),
                                        /*blocking=*/false);
    distributed::EnqueueWriteMeshBuffer(cq, b_buf, sscad::tilize(sscad::pad_to_tiles(b, m, n), mp, np),
                                        /*blocking=*/false);

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()),
                         std::move(program));

    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

    std::vector<bfloat16> c_tiled;
    distributed::EnqueueReadMeshBuffer(cq, c_tiled, c_buf, /*blocking=*/true);

    const std::vector<bfloat16> c = sscad::unpad(sscad::untilize(c_tiled, mp, np), m, n);
    const sscad::CompareResult result = sscad::compare(c, sscad::reference::eltwise_add(a, b));
    sscad::print_compare(result);

    return sscad::report(sscad::close_device(mesh_device) && result.pass);
}
