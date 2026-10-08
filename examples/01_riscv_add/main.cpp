// SPDX-License-Identifier: Apache-2.0
//
// Exemplo 01 — soma de dois inteiros no RISC-V de movimentação de dados.
//
// Os processadores que executam os kernels de movimentação de dados são CPUs
// RISC-V comuns (pequenas). Este exemplo mostra que eles executam C++ normal:
// o kernel lê dois inteiros da DRAM, soma-os e escreve o resultado de volta.
// As limitações desses núcleos estão descritas no kernel e no README.

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "compare.hpp"
#include "device.hpp"

using namespace tt::tt_metal;

int main(int argc, char **argv) {
    // Uso: ex01_riscv_add [a] [b]   (aceita decimal ou 0x...)
    const uint32_t a = argc > 1 ? std::stoul(argv[1], nullptr, 0) : 14;
    const uint32_t b = argc > 2 ? std::stoul(argv[2], nullptr, 0) : 7;
    fmt::print("Exemplo 01: {} + {} no RISC-V de movimentação de dados\n", a, b);

    if (std::getenv("TT_METAL_DPRINT_CORES") == nullptr) {
        fmt::print("Dica: execute `export TT_METAL_DPRINT_CORES=0,0` para ver a "
                   "mensagem impressa pelo kernel.\n");
    }

    auto mesh_device = sscad::open_device();
    distributed::MeshCommandQueue &cq = mesh_device->mesh_command_queue();
    constexpr CoreCoord core = {0, 0};

    // ANCHOR: host_buffers
    // Aqui a página é um único inteiro (4 bytes), e não um tile. O alocador
    // ainda alinha cada página ao alinhamento da memória (32 B na DRAM e 16 B
    // na L1 do Wormhole), então cada buffer ocupa na prática um bloco alinhado.
    constexpr uint32_t bytes = sizeof(uint32_t);
    distributed::DeviceLocalBufferConfig dram_config{.page_size = bytes,
                                                     .buffer_type = BufferType::DRAM};
    distributed::DeviceLocalBufferConfig l1_config{.page_size = bytes,
                                                   .buffer_type = BufferType::L1};
    distributed::ReplicatedBufferConfig size{.size = bytes};

    // Operandos e resultado na DRAM...
    auto a_dram = distributed::MeshBuffer::create(size, dram_config, mesh_device.get());
    auto b_dram = distributed::MeshBuffer::create(size, dram_config, mesh_device.get());
    auto c_dram = distributed::MeshBuffer::create(size, dram_config, mesh_device.get());
    // ...e suas cópias de trabalho na L1, onde o RISC-V consegue acessá-los.
    auto a_l1 = distributed::MeshBuffer::create(size, l1_config, mesh_device.get());
    auto b_l1 = distributed::MeshBuffer::create(size, l1_config, mesh_device.get());
    auto c_l1 = distributed::MeshBuffer::create(size, l1_config, mesh_device.get());
    // ANCHOR_END: host_buffers

    Program program = CreateProgram();

    // ANCHOR: host_kernel
    std::vector<uint32_t> compile_args;
    TensorAccessorArgs(*a_dram).append_to(compile_args);
    TensorAccessorArgs(*b_dram).append_to(compile_args);
    TensorAccessorArgs(*c_dram).append_to(compile_args);

    KernelHandle kernel = CreateKernel(
        program, KERNEL_DIR "dataflow/riscv_add.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc = NOC::RISCV_0_default,
                           .compile_args = compile_args});
    // ANCHOR_END: host_kernel

    // ANCHOR: host_runtime_args
    SetRuntimeArgs(program, kernel, core,
                   {static_cast<uint32_t>(a_dram->address()),
                    static_cast<uint32_t>(b_dram->address()),
                    static_cast<uint32_t>(c_dram->address()),
                    static_cast<uint32_t>(a_l1->address()),
                    static_cast<uint32_t>(b_l1->address()),
                    static_cast<uint32_t>(c_l1->address())});
    // ANCHOR_END: host_runtime_args

    // ANCHOR: host_enqueue
    distributed::EnqueueWriteMeshBuffer(cq, a_dram, std::vector<uint32_t>{a}, /*blocking=*/false);
    distributed::EnqueueWriteMeshBuffer(cq, b_dram, std::vector<uint32_t>{b}, /*blocking=*/false);

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()),
                         std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

    std::vector<uint32_t> c;
    distributed::EnqueueReadMeshBuffer(cq, c, c_dram, /*blocking=*/true);
    // ANCHOR_END: host_enqueue

    // Aritmética de uint32_t no host também dá a volta em 2^32, como no RISC-V.
    const uint32_t expected = a + b;
    const bool pass = c.size() == 1 && c[0] == expected;
    fmt::print("Resultado: {} (esperado {})\n", c.empty() ? 0 : c[0], expected);

    return sscad::report(sscad::close_device(mesh_device) && pass);
}
