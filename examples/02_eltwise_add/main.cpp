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
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>

#include "compare.hpp"
#include "reference.hpp"

using namespace tt::tt_metal;

int main() {
    // As dimensões são múltiplos de 32, de modo que as matrizes se dividem
    // exatamente em tiles 32x32.
    constexpr uint32_t m = 64;
    constexpr uint32_t n = 64;
    static_assert(m % tt::constants::TILE_HEIGHT == 0 && n % tt::constants::TILE_WIDTH == 0);
    constexpr uint32_t num_tiles = (m / tt::constants::TILE_HEIGHT) * (n / tt::constants::TILE_WIDTH);
    constexpr uint32_t tile_bytes = sizeof(bfloat16) * tt::constants::TILE_HW;
    fmt::print("Exemplo 02: C = A + B com A, B, C bfloat16 de {}x{} em 1 núcleo\n", m, n);

    // A e B aleatórios com semente fixa, para que a execução seja reprodutível.
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
    constexpr CoreCoord core = {0, 0};

    // ANCHOR: host_buffers
    // Três buffers interleaved na DRAM, uma página = um tile bfloat16 (2048 B).
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = tile_bytes, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig size{.size = num_tiles * tile_bytes};

    std::shared_ptr<distributed::MeshBuffer> a_buf =
        distributed::MeshBuffer::create(size, dram_config, mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> b_buf =
        distributed::MeshBuffer::create(size, dram_config, mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> c_buf =
        distributed::MeshBuffer::create(size, dram_config, mesh_device.get());
    // ANCHOR_END: host_buffers

    Program program = CreateProgram();

    // ANCHOR: host_cbs
    // Circular buffers (CBs) são filas de tiles na L1 que ligam os kernels:
    // um produtor reserva espaço e publica tiles; um consumidor espera por
    // eles e os libera. Com 2 tiles por CB (double buffering), o produtor
    // pode preencher um tile enquanto o consumidor processa o outro, e a
    // leitura, a soma e a escrita se sobrepõem no tempo.
    constexpr uint32_t tiles_per_cb = 2;

    CircularBufferConfig cb_a_config =
        CircularBufferConfig(tiles_per_cb * tile_bytes, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_0, tile_bytes);
    CreateCircularBuffer(program, core, cb_a_config);

    CircularBufferConfig cb_b_config =
        CircularBufferConfig(tiles_per_cb * tile_bytes, {{tt::CBIndex::c_1, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_1, tile_bytes);
    CreateCircularBuffer(program, core, cb_b_config);

    CircularBufferConfig cb_c_config =
        CircularBufferConfig(tiles_per_cb * tile_bytes, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_16, tile_bytes);
    CreateCircularBuffer(program, core, cb_c_config);
    // ANCHOR_END: host_cbs

    // ANCHOR: host_kernels
    std::vector<uint32_t> reader_args;
    TensorAccessorArgs(*a_buf).append_to(reader_args);
    TensorAccessorArgs(*b_buf).append_to(reader_args);
    std::vector<uint32_t> writer_args;
    TensorAccessorArgs(*c_buf).append_to(writer_args);

    // Defina SSCAD_DPRINT=1 para compilar o DPRINT de depuração do reader
    // (também é preciso TT_METAL_DPRINT_CORES=0,0).
    std::map<std::string, std::string> reader_defines;
    if (std::getenv("SSCAD_DPRINT") != nullptr) {
        reader_defines["SSCAD_DPRINT"] = "1";
    }

    KernelHandle reader = CreateKernel(
        program, KERNEL_DIR "dataflow/reader.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc = NOC::RISCV_0_default,
                           .compile_args = reader_args,
                           .defines = reader_defines});
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
    // As matrizes vão para o dispositivo no layout em tiles: sequência de
    // tiles 32x32, cada um dividido em 4 faces 16x16, que é o formato
    // esperado pelas unidades de cálculo do Tensix.
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
    // ANCHOR_END: host_enqueue

    const std::vector<bfloat16> c = convert_layout<bfloat16>(
        c_tiled, {m, n}, TensorLayoutType::TILED_NFACES, TensorLayoutType::LIN_ROW_MAJOR);
    const sscad::CompareResult result = sscad::compare(c, sscad::reference::eltwise_add(a, b));
    sscad::print_compare(result);

    const bool closed = mesh_device->close();
    return sscad::report(closed && result.pass);
}
