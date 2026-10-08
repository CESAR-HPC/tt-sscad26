// SPDX-License-Identifier: Apache-2.0
//
// Exemplo 02 — soma elemento a elemento C = A + B em um núcleo.
//
// Menor programa que mostra o modelo de programação completo:
//
//   host -> DRAM -> reader -> CB -> compute -> CB -> writer -> DRAM -> host
//
// Três kernels rodam ao mesmo tempo no núcleo {0,0}: o reader (RISC-V 0)
// traz tiles de A e B da DRAM para circular buffers, o compute soma-os nos
// motores do Tensix e o writer (RISC-V 1) leva o resultado de volta à DRAM.

#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "compare.hpp"
#include "data_format.hpp"
#include "device.hpp"
#include "layout.hpp"
#include "reference.hpp"

using namespace tt::tt_metal;

int main(int argc, char **argv) {
    // Uso: ex02_eltwise_add [M] [N]
    const uint32_t m = argc > 1 ? std::stoul(argv[1]) : 64;
    const uint32_t n = argc > 2 ? std::stoul(argv[2]) : 64;
    fmt::print("Exemplo 02: C = A + B com A, B, C bfloat16 de {}x{} em 1 núcleo\n", m, n);

    // Dimensões completadas até múltiplos de 32, pois o dispositivo opera em tiles.
    const uint32_t mp = sscad::round_up_to_tile(m);
    const uint32_t np = sscad::round_up_to_tile(n);
    const uint32_t num_tiles = (mp / sscad::TILE) * (np / sscad::TILE);
    constexpr uint32_t tile_bytes = sscad::tile_size_bytes<bfloat16>();

    // A e B aleatórios com semente fixa, para que a execução seja reprodutível.
    std::vector<bfloat16> a(m * n), b(m * n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : a) v = bfloat16(dist(rng));
    for (auto &v : b) v = bfloat16(dist(rng));

    auto mesh_device = sscad::open_device();
    distributed::MeshCommandQueue &cq = mesh_device->mesh_command_queue();
    constexpr CoreCoord core = {0, 0};

    // ANCHOR: host_buffers
    // Três buffers interleaved na DRAM, uma página = um tile bfloat16 (2048 B).
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = tile_bytes, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig size{.size = num_tiles * tile_bytes};

    auto a_buf = distributed::MeshBuffer::create(size, dram_config, mesh_device.get());
    auto b_buf = distributed::MeshBuffer::create(size, dram_config, mesh_device.get());
    auto c_buf = distributed::MeshBuffer::create(size, dram_config, mesh_device.get());
    // ANCHOR_END: host_buffers

    Program program = CreateProgram();

    // ANCHOR: host_cbs
    // Circular buffers (CBs) são filas de tiles na L1 que ligam os kernels:
    // um produtor reserva espaço e publica tiles; um consumidor espera por
    // eles e os libera. Com 2 tiles por CB (double buffering), o produtor
    // pode preencher um tile enquanto o consumidor processa o outro, e a
    // leitura, a soma e a escrita se sobrepõem no tempo.
    constexpr uint32_t tiles_per_cb = 2;
    auto make_cb = [&](tt::CBIndex index) {
        CircularBufferConfig config =
            CircularBufferConfig(tiles_per_cb * tile_bytes,
                                 {{index, sscad::data_format_of<bfloat16>()}})
                .set_page_size(index, tile_bytes);
        CreateCircularBuffer(program, core, config);
    };
    make_cb(tt::CBIndex::c_0);  // A
    make_cb(tt::CBIndex::c_1);  // B
    make_cb(tt::CBIndex::c_16); // C
    // ANCHOR_END: host_cbs

    // ANCHOR: host_kernels
    std::vector<uint32_t> reader_args;
    TensorAccessorArgs(*a_buf).append_to(reader_args);
    TensorAccessorArgs(*b_buf).append_to(reader_args);
    std::vector<uint32_t> writer_args;
    TensorAccessorArgs(*c_buf).append_to(writer_args);

    // Defina SSCAD_DPRINT=1 para compilar o DPRINT de depuração do reader
    // (também é preciso TT_METAL_DPRINT_CORES=0,0).
    std::map<std::string, std::string> defines;
    if (std::getenv("SSCAD_DPRINT") != nullptr) {
        defines["SSCAD_DPRINT"] = "1";
    }

    KernelHandle reader = CreateKernel(
        program, KERNEL_DIR "dataflow/reader.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc = NOC::RISCV_0_default,
                           .compile_args = reader_args,
                           .defines = defines});
    KernelHandle writer = CreateKernel(
        program, KERNEL_DIR "dataflow/writer.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                           .noc = NOC::RISCV_1_default,
                           .compile_args = writer_args});
    KernelHandle compute =
        CreateKernel(program, KERNEL_DIR "compute/compute.cpp", core, ComputeConfig{});
    // ANCHOR_END: host_kernels

    // ANCHOR: host_runtime_args
    SetRuntimeArgs(program, reader, core,
                   {static_cast<uint32_t>(a_buf->address()),
                    static_cast<uint32_t>(b_buf->address()), num_tiles});
    SetRuntimeArgs(program, compute, core, {num_tiles});
    SetRuntimeArgs(program, writer, core,
                   {static_cast<uint32_t>(c_buf->address()), num_tiles});
    // ANCHOR_END: host_runtime_args

    // ANCHOR: host_enqueue
    // As matrizes vão para o dispositivo no layout em tiles.
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
    // ANCHOR_END: host_enqueue

    const std::vector<bfloat16> c = sscad::unpad(sscad::untilize(c_tiled, mp, np), m, n);
    const sscad::CompareResult result = sscad::compare(c, sscad::reference::eltwise_add(a, b));
    sscad::print_compare(result);

    return sscad::report(sscad::close_device(mesh_device) && result.pass);
}
