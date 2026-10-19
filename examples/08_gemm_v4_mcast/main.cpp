// SPDX-License-Identifier: Apache-2.0
//
// Exemplo 08 — GEMM v4 (multicast): C = A B.
//
// O compute, o writer e a divisão em blocos são os do exemplo 07. A diferença
// está no reader: no exemplo 07 os núcleos de uma mesma linha da grade leem
// da DRAM o mesmo bloco de A, e os de uma mesma coluna, o mesmo bloco de B.
// Aqui só o núcleo da coluna 0 lê o bloco de A da linha e o envia por
// multicast aos demais núcleos da linha; só o da linha 0 lê o bloco de B da
// coluna e o envia aos demais da coluna. Cada bloco sai da DRAM uma vez, em
// vez de num_blocks_x (A) ou num_blocks_y (B) vezes.
//
// Emissor e receptores se sincronizam com semáforos em L1: o receptor avisa
// que o seu CB tem espaço (incrementa o semáforo do emissor), e o emissor,
// depois de enviar o bloco, marca o semáforo de cada receptor como VALID.

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
#include <hostdevcommon/common_values.hpp>

#include "compare.hpp"
#include "reference.hpp"
#include "timer.hpp"

using namespace tt::tt_metal;

int main() {
    // Os mesmos tamanhos dos exemplos 05–09, para comparar os tempos. Com
    // blocos de 2x2 tiles, os 18x16 tiles de C formam uma grade de 8x9 blocos:
    // um por núcleo, ocupando os 72 núcleos do simulador.
    constexpr uint32_t m = 576;
    constexpr uint32_t n = 512;
    constexpr uint32_t k = 384;
    static_assert(m % tt::constants::TILE_HEIGHT == 0 &&
                  n % tt::constants::TILE_WIDTH == 0 &&
                  k % tt::constants::TILE_WIDTH == 0);
    constexpr uint32_t Mt = m / tt::constants::TILE_HEIGHT;
    constexpr uint32_t Nt = n / tt::constants::TILE_WIDTH;
    constexpr uint32_t Kt = k / tt::constants::TILE_WIDTH;

    // ANCHOR: host_blocking
    // Bloco de C por núcleo: per_core_M x per_core_N tiles. Passo em K:
    // in0_block_w tiles. Sub-bloco no Dst: out_subblock_h x out_subblock_w
    // tiles; com fp32_dest_acc_en metade do Dst guarda 4 tiles FP32 (a outra
    // metade fica com o packer), então o sub-bloco tem no máximo 4 tiles.
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
    // Pelo menos um receptor por linha e por coluna da grade.
    static_assert(num_blocks_x > 1 && num_blocks_y > 1);
    // ANCHOR_END: host_blocking

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

    // ANCHOR: host_cores
    // Um bloco de C por núcleo: o bloco (by, bx) fica no núcleo lógico
    // (x = bx, y = by). A grade de núcleos tem a forma da grade de blocos.
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    if (num_blocks_x > grid.x || num_blocks_y > grid.y) {
        fmt::print("A grade de blocos {}x{} não cabe na grade de núcleos {}x{}\n",
                   num_blocks_x, num_blocks_y, grid.x, grid.y);
        return 1;
    }
    const tt::tt_metal::CoreRange all_cores({0, 0},
                                            {num_blocks_x - 1, num_blocks_y - 1});
    // Coluna 0: emissores de A. As demais colunas recebem A.
    const tt::tt_metal::CoreRange left_column({0, 0}, {0, num_blocks_y - 1});
    const tt::tt_metal::CoreRange other_columns(
        {1, 0}, {num_blocks_x - 1, num_blocks_y - 1});
    constexpr uint32_t num_cores = num_blocks_x * num_blocks_y;
    // ANCHOR_END: host_cores

    fmt::print("Exemplo 08: C = A B com A {}x{}, B {}x{} bfloat16 e C float em "
               "{} de {}x{} núcleos\n",
               m, k, k, n, num_cores, grid.x, grid.y);
    fmt::print("  bloco de {}x{} tiles por núcleo, passo de {} tiles em K, "
               "sub-blocos de {}x{}\n",
               per_core_M, per_core_N, in0_block_w, out_subblock_h,
               out_subblock_w);

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
    // A e B: dois blocos cada (double buffering): o reader traz o bloco do
    // próximo passo de K enquanto o compute usa o atual.
    constexpr uint32_t a_block_tiles = per_core_M * in0_block_w;
    constexpr uint32_t b_block_tiles = in0_block_w * per_core_N;
    constexpr uint32_t out_block_tiles = per_core_M * per_core_N;

    CircularBufferConfig cb_a_config =
        CircularBufferConfig(2 * a_block_tiles * in_tile_bytes,
                             {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_0, in_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_a_config);

    CircularBufferConfig cb_b_config =
        CircularBufferConfig(2 * b_block_tiles * in_tile_bytes,
                             {{tt::CBIndex::c_1, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_1, in_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_b_config);

    // C: o bloco de saída inteiro.
    CircularBufferConfig cb_c_config =
        CircularBufferConfig(out_block_tiles * out_tile_bytes,
                             {{tt::CBIndex::c_16, tt::DataFormat::Float32}})
            .set_page_size(tt::CBIndex::c_16, out_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_c_config);

    // Somas parciais entre passos de K, em float. Ao serem recarregadas no
    // Dst elas passam por SrcA, que guarda 19 bits (TF32): por isso o erro é
    // um pouco maior que o dos exemplos 05 e 06, que nunca tiram a soma do
    // Dst. Ainda fica bem abaixo da tolerância de bfloat16.
    CircularBufferConfig cb_partial_config =
        CircularBufferConfig(out_block_tiles * out_tile_bytes,
                             {{tt::CBIndex::c_24, tt::DataFormat::Float32}})
            .set_page_size(tt::CBIndex::c_24, out_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_partial_config);
    // ANCHOR_END: host_cbs

    std::vector<uint32_t> reader_args;
    TensorAccessorArgs(*a_buf).append_to(reader_args);
    TensorAccessorArgs(*b_buf).append_to(reader_args);
    std::vector<uint32_t> writer_args;
    TensorAccessorArgs(*c_buf).append_to(writer_args);
    // A forma dos blocos é a mesma em todos os núcleos: argumentos de
    // compilação, que permitem ao compilador desenrolar os laços do compute.
    const std::vector<uint32_t> compute_args = {
        in0_block_w,    num_k_blocks,   per_core_M,
        per_core_N,     out_subblock_h, out_subblock_w};

    // ANCHOR: host_kernels
    // O mesmo reader em dois grupos de núcleos, com NoCs diferentes: na
    // coluna 0 o reader usa a NoC 0 (o multicast de A vai para a direita) e o
    // writer, a NoC 1; nas demais colunas, o contrário. Assim reader e
    // writer de um núcleo não disputam a mesma NoC.
    KernelHandle reader_left = CreateKernel(
        program, KERNEL_DIR "dataflow/reader.cpp", left_column,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc = NOC::NOC_0,
                           .compile_args = reader_args});
    KernelHandle reader_other = CreateKernel(
        program, KERNEL_DIR "dataflow/reader.cpp", other_columns,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc = NOC::NOC_1,
                           .compile_args = reader_args});
    KernelHandle writer_left = CreateKernel(
        program, KERNEL_DIR "dataflow/writer.cpp", left_column,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                           .noc = NOC::NOC_1,
                           .compile_args = writer_args});
    KernelHandle writer_other = CreateKernel(
        program, KERNEL_DIR "dataflow/writer.cpp", other_columns,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                           .noc = NOC::NOC_0,
                           .compile_args = writer_args});
    CreateKernel(program, KERNEL_DIR "compute/compute.cpp", all_cores,
                 ComputeConfig{.math_fidelity =
                                   tt::tt_metal::MathFidelity::HiFi4,
                               .fp32_dest_acc_en = true,
                               .compile_args = compute_args});
    // ANCHOR_END: host_kernels

    // ANCHOR: host_semaphores
    // Quatro semáforos em todos os núcleos (o mesmo id vale em todos):
    //   *_ready: no emissor, conta os receptores com espaço no CB;
    //   *_valid: no receptor, vira VALID quando o bloco chegou.
    const uint32_t a_ready_sem = CreateSemaphore(program, all_cores, INVALID);
    const uint32_t a_valid_sem = CreateSemaphore(program, all_cores, INVALID);
    const uint32_t b_ready_sem = CreateSemaphore(program, all_cores, INVALID);
    const uint32_t b_valid_sem = CreateSemaphore(program, all_cores, INVALID);
    // ANCHOR_END: host_semaphores

    // ANCHOR: host_runtime_args
    // O kernel endereça os outros núcleos por coordenadas da NoC, e não pelas
    // coordenadas lógicas da grade: worker_core_from_logical_core converte.
    // Um retângulo de multicast vai do canto "start" ao "end" no sentido em
    // que a NoC anda: na NoC 0 (x e y crescentes), do menor para o maior; na
    // NoC 1 (x e y decrescentes), do maior para o menor.
    for (uint32_t by = 0; by < num_blocks_y; by++) {
        for (uint32_t bx = 0; bx < num_blocks_x; bx++) {
            const CoreCoord core = {bx, by};
            const bool on_noc_0 = bx == 0;

            // A: emissor na coluna 0, receptores nas colunas 1..X-1. Só a
            // coluna 0 envia A, e ela usa a NoC 0: retângulo first -> last.
            const CoreCoord a_sender =
                mesh_device->worker_core_from_logical_core({0, by});
            const CoreCoord a_first =
                mesh_device->worker_core_from_logical_core({1, by});
            const CoreCoord a_last = mesh_device->worker_core_from_logical_core(
                {num_blocks_x - 1, by});

            // B: emissor na linha 0, receptores nas linhas 1..Y-1.
            const CoreCoord b_sender =
                mesh_device->worker_core_from_logical_core({bx, 0});
            const CoreCoord b_first =
                mesh_device->worker_core_from_logical_core({bx, 1});
            const CoreCoord b_last = mesh_device->worker_core_from_logical_core(
                {bx, num_blocks_y - 1});
            const CoreCoord b_start = on_noc_0 ? b_first : b_last;
            const CoreCoord b_end = on_noc_0 ? b_last : b_first;

            const std::vector<uint32_t> reader_runtime_args = {
                static_cast<uint32_t>(a_buf->address()),
                static_cast<uint32_t>(b_buf->address()),
                Kt,
                Nt,
                by,
                bx,
                per_core_M,
                per_core_N,
                in0_block_w,
                static_cast<uint32_t>(a_sender.x),
                static_cast<uint32_t>(a_sender.y),
                static_cast<uint32_t>(a_first.x),
                static_cast<uint32_t>(a_first.y),
                static_cast<uint32_t>(a_last.x),
                static_cast<uint32_t>(a_last.y),
                num_blocks_x - 1,
                static_cast<uint32_t>(b_sender.x),
                static_cast<uint32_t>(b_sender.y),
                static_cast<uint32_t>(b_start.x),
                static_cast<uint32_t>(b_start.y),
                static_cast<uint32_t>(b_end.x),
                static_cast<uint32_t>(b_end.y),
                num_blocks_y - 1,
                a_ready_sem,
                a_valid_sem,
                b_ready_sem,
                b_valid_sem};
            const std::vector<uint32_t> writer_runtime_args = {
                static_cast<uint32_t>(c_buf->address()),
                Nt,
                by,
                bx,
                per_core_M,
                per_core_N,
                out_subblock_h,
                out_subblock_w};

            SetRuntimeArgs(program, on_noc_0 ? reader_left : reader_other,
                           core, reader_runtime_args);
            SetRuntimeArgs(program, on_noc_0 ? writer_left : writer_other,
                           core, writer_runtime_args);
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
    fmt::print("08_gemm_v4_mcast,{},{},{},bf16-fp32,{},{:.3f},{:.3f},{:.3e}\n",
               m, n, k, num_cores, time_ms, gflops, result.max_rel_err);

    const bool closed = mesh_device->close();
    return sscad::report(closed && result.pass);
}
