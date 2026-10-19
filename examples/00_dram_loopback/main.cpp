// Exemplo 00 — DRAM loopback.
//
// Primeiro contato com o TT-Metalium: o host coloca dados na DRAM do
// dispositivo, um kernel de movimentação de dados copia-os, tile a tile, para
// outro buffer da DRAM passando pela SRAM (L1) do núcleo, e o host lê o
// resultado de volta. Não há kernel de computação nem circular buffers.
//
//   host --(PCIe)--> DRAM (entrada) --(NoC)--> L1 --(NoC)--> DRAM (saída)
//   --(PCIe)--> host

#include <bit>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <fmt/core.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "compare.hpp"

using namespace tt::tt_metal;

int main() {
    constexpr uint32_t num_tiles = 8;
    fmt::print("Exemplo 00: cópia DRAM -> L1 -> DRAM de {} tiles bfloat16\n",
               num_tiles);

    // Um MeshDevice 1x1, isto é, um único chip. A mesma API escala para malhas
    // com vários chips, mas no minicurso trabalhamos sempre com um só.
    std::shared_ptr<distributed::MeshDevice> mesh_device =
        distributed::MeshDevice::create_unit_mesh(0);

    // Toda comunicação com o dispositivo (escritas, leituras e execução de
    // programas) passa pela fila de comandos, que executa em ordem.
    distributed::MeshCommandQueue &cq = mesh_device->mesh_command_queue();

    // Todo o trabalho deste exemplo é feito por um único núcleo Tensix.
    constexpr CoreCoord core = {0, 0};

    // Um tile tem 32x32 elementos; em bfloat16 isso dá 2048 bytes.
    constexpr uint32_t tile_bytes = sizeof(bfloat16) * tt::constants::TILE_HW;
    constexpr uint32_t num_elements = num_tiles * tt::constants::TILE_HW;

    // ANCHOR: host_buffers
    // Buffers "interleaved": o buffer é dividido em páginas (aqui, 1 página =
    // 1 tile) e as páginas são distribuídas em round-robin entre os bancos de
    // DRAM. O kernel não precisa saber em que banco está cada página: o
    // TensorAccessor faz essa conta a partir do índice da página.
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = tile_bytes, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig dram_size{.size =
                                                      num_tiles * tile_bytes};

    std::shared_ptr<distributed::MeshBuffer> input =
        distributed::MeshBuffer::create(dram_size, dram_config,
                                        mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> output =
        distributed::MeshBuffer::create(dram_size, dram_config,
                                        mesh_device.get());

    // Um núcleo Tensix não lê a DRAM diretamente: todo dado passa pela L1
    // (a SRAM local do núcleo, ~1.5 MB no Wormhole) através da NoC. Por isso
    // alocamos um buffer de um tile na L1, que o kernel usa como área de
    // trabalho para cada cópia.
    distributed::DeviceLocalBufferConfig l1_config{
        .page_size = tile_bytes, .buffer_type = BufferType::L1};
    distributed::ReplicatedBufferConfig l1_size{.size = tile_bytes};

    std::shared_ptr<distributed::MeshBuffer> scratch =
        distributed::MeshBuffer::create(l1_size, l1_config, mesh_device.get());
    // ANCHOR_END: host_buffers

    // Os dados são bytes opacos para a cópia, então não é preciso convertê-los
    // para o layout em tiles: basta que a saída seja idêntica à entrada.
    std::vector<bfloat16> input_vec(num_elements);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    for (bfloat16 &value : input_vec) {
        value = bfloat16(dist(rng));
    }

    Program program = CreateProgram();

    // ANCHOR: host_kernel
    // Os argumentos de compilação descrevem como os buffers estão
    // distribuídos (interleaved, DRAM, tamanho de página). Eles são lidos
    // pelo TensorAccessorArgs no kernel.
    std::vector<uint32_t> compile_args;
    TensorAccessorArgs(*input).append_to(compile_args);
    TensorAccessorArgs(*output).append_to(compile_args);

    // Kernel de movimentação de dados no RISC-V 0 (BRISC), usando a NoC 0.
    KernelHandle loopback = CreateKernel(
        program, KERNEL_DIR "dataflow/loopback.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc = NOC::RISCV_0_default,
                           .compile_args = compile_args});
    // ANCHOR_END: host_kernel

    // ANCHOR: host_runtime_args
    // Argumentos de execução: endereços dos buffers e número de tiles.
    SetRuntimeArgs(program, loopback, core,
                   {static_cast<uint32_t>(scratch->address()),
                    static_cast<uint32_t>(input->address()),
                    static_cast<uint32_t>(output->address()), num_tiles});
    // ANCHOR_END: host_runtime_args

    // ANCHOR: host_enqueue
    // A fila de comandos executa em ordem: escrita -> programa -> leitura.
    distributed::EnqueueWriteMeshBuffer(cq, input, input_vec,
                                        /*blocking=*/false);

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()),
                         std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

    std::vector<bfloat16> output_vec;
    distributed::EnqueueReadMeshBuffer(cq, output_vec, output,
                                       /*blocking=*/true);
    // ANCHOR_END: host_enqueue

    // Uma cópia deve ser exata: comparamos bit a bit, sem tolerância.
    const bool same_size = output_vec.size() == input_vec.size();
    uint32_t mismatches = 0;
    for (uint32_t i = 0; same_size && i < num_elements; i++) {
        if (std::bit_cast<uint16_t>(input_vec[i]) !=
            std::bit_cast<uint16_t>(output_vec[i])) {
            mismatches++;
        }
    }
    fmt::print("Elementos diferentes: {} de {}\n", mismatches, num_elements);

    const bool closed = mesh_device->close();
    return sscad::report(closed && same_size && mismatches == 0);
}
