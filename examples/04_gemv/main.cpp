// SPDX-License-Identifier: Apache-2.0
//
// Exemplo 04 — GEMV multicore: y = alpha * A x + beta * y.
//
// A e x são bfloat16; y, alpha e beta são float (FP32), e o produto A x é
// acumulado em FP32 no registrador Dst.
//
// O motor de matrizes só multiplica tiles 32x32, então os vetores precisam
// virar matrizes de tiles (padding): cada vetor ocupa a coluna 0 dos seus
// tiles e o resto é zero. Assim, o produto de um tile de A por um tile de x
// deixa na coluna 0 do tile de saída a contribuição daquele bloco para y.
//
// Cada núcleo recebe uma faixa de linhas de tiles de A (uma linha de tiles
// produz 32 elementos de y).

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

// ANCHOR: host_padding
// Preenchimento dos vetores. Nenhuma das duas funções usa o TT-Metalium:
// elas só montam, em row-major, as matrizes que serão convertidas para o
// layout de tiles.

// Vetor x de tamanho rows -> matriz rows x cols com x[i] na coluna 0 de cada
// coluna de tiles: padded[i][32 * j] = x[i]. O resto é zero.
template <typename T>
std::vector<T> pad_vector_to_matrix(const std::vector<T> &vec, uint32_t rows,
                                    uint32_t cols) {
    std::vector<T> padded(rows * cols, static_cast<T>(0));
    const uint32_t tile_cols = cols / tt::constants::TILE_WIDTH;
    for (uint32_t i = 0; i < rows; i++) {
        for (uint32_t j = 0; j < tile_cols; j++) {
            padded[i * cols + tt::constants::TILE_WIDTH * j] = vec[i];
        }
    }
    return padded;
}

// Vetor y de tamanho rows -> matriz rows x cols com y[i] apenas na coluna 0:
// padded[i][0] = y[i]. O resto é zero.
template <typename T>
std::vector<T> pad_vector_with_zeros(const std::vector<T> &vec, uint32_t rows,
                                     uint32_t cols) {
    std::vector<T> padded(rows * cols, static_cast<T>(0));
    for (uint32_t i = 0; i < rows; i++) {
        padded[i * cols] = vec[i];
    }
    return padded;
}
// ANCHOR_END: host_padding

int main() {
    // A é m x n. m != n para que os índices de linha e de coluna não se
    // confundam.
    constexpr uint32_t m = 512;
    constexpr uint32_t n = 256;
    static_assert(m % tt::constants::TILE_HEIGHT == 0 &&
                  n % tt::constants::TILE_WIDTH == 0);
    constexpr uint32_t Mt =
        m / tt::constants::TILE_HEIGHT; // linhas de tiles de A (e de y)
    constexpr uint32_t Nt =
        n / tt::constants::TILE_WIDTH; // colunas de tiles de A
    constexpr float alpha = 2.0f;
    constexpr float beta = 0.5f;

    // Tiles de entrada em bfloat16, tiles de y/alpha/beta/saída em float.
    constexpr uint32_t in_tile_bytes =
        sizeof(bfloat16) * tt::constants::TILE_HW;
    constexpr uint32_t acc_tile_bytes = sizeof(float) * tt::constants::TILE_HW;

    std::vector<bfloat16> a(m * n);
    std::vector<bfloat16> x(n);
    std::vector<float> y(m);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (bfloat16 &value : a) {
        value = bfloat16(dist(rng));
    }
    for (bfloat16 &value : x) {
        value = bfloat16(dist(rng));
    }
    for (float &value : y) {
        value = dist(rng);
    }

    // ANCHOR: host_layout
    // A (m x n) é convertida diretamente. x vira uma matriz n x m: o tile
    // (nt, mt) contém x[32 nt : 32 nt + 32] na coluna 0, e o reader lê o tile
    // (nt, mt) para a linha de tiles mt. y vira uma matriz m x m com y na
    // coluna 0; só a primeira coluna de tiles (tiles (mt, 0)) é usada.
    // alpha e beta viram tiles cheios do mesmo valor, para que o SFPU possa
    // multiplicá-los elemento a elemento.
    const std::vector<bfloat16> x_padded = pad_vector_to_matrix(x, n, m);
    const std::vector<float> y_padded = pad_vector_with_zeros(y, m, m);
    const std::vector<float> alpha_tile(tt::constants::TILE_HW, alpha);
    const std::vector<float> beta_tile(tt::constants::TILE_HW, beta);

    const std::vector<bfloat16> a_tiled =
        convert_layout<bfloat16>(a, {m, n}, TensorLayoutType::LIN_ROW_MAJOR,
                                 TensorLayoutType::TILED_NFACES);
    const std::vector<bfloat16> x_tiled = convert_layout<bfloat16>(
        x_padded, {n, m}, TensorLayoutType::LIN_ROW_MAJOR,
        TensorLayoutType::TILED_NFACES);
    const std::vector<float> y_tiled =
        convert_layout<float>(y_padded, {m, m}, TensorLayoutType::LIN_ROW_MAJOR,
                              TensorLayoutType::TILED_NFACES);
    const std::vector<float> alpha_tiled = convert_layout<float>(
        alpha_tile, {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
        TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES);
    const std::vector<float> beta_tiled = convert_layout<float>(
        beta_tile, {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
        TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES);
    // ANCHOR_END: host_layout

    std::shared_ptr<distributed::MeshDevice> mesh_device =
        distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue &cq = mesh_device->mesh_command_queue();

    // ANCHOR: host_split
    // A unidade de trabalho é uma linha de tiles de A: ela produz um tile de
    // y inteiro e não depende das outras linhas.
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();

    uint32_t num_cores = 0;
    tt::tt_metal::CoreRangeSet all_cores;
    tt::tt_metal::CoreRangeSet group_1;
    tt::tt_metal::CoreRangeSet group_2;
    uint32_t rows_per_core_1 = 0;
    uint32_t rows_per_core_2 = 0;
    std::tie(num_cores, all_cores, group_1, group_2, rows_per_core_1,
             rows_per_core_2) = split_work_to_cores(grid, Mt);
    // ANCHOR_END: host_split

    fmt::print("Exemplo 04: y = {} * A x + {} * y com A {}x{} bfloat16 e y "
               "float em {} de {}x{} núcleos\n",
               alpha, beta, m, n, num_cores, grid.x, grid.y);

    // ANCHOR: host_buffers
    distributed::DeviceLocalBufferConfig in_dram_config{
        .page_size = in_tile_bytes, .buffer_type = BufferType::DRAM};
    distributed::DeviceLocalBufferConfig acc_dram_config{
        .page_size = acc_tile_bytes, .buffer_type = BufferType::DRAM};

    distributed::ReplicatedBufferConfig a_size{.size = Mt * Nt * in_tile_bytes};
    distributed::ReplicatedBufferConfig x_size{.size = Nt * Mt * in_tile_bytes};
    distributed::ReplicatedBufferConfig y_size{.size =
                                                   Mt * Mt * acc_tile_bytes};
    distributed::ReplicatedBufferConfig scalar_size{.size = acc_tile_bytes};

    std::shared_ptr<distributed::MeshBuffer> a_buf =
        distributed::MeshBuffer::create(a_size, in_dram_config,
                                        mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> x_buf =
        distributed::MeshBuffer::create(x_size, in_dram_config,
                                        mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> y_buf =
        distributed::MeshBuffer::create(y_size, acc_dram_config,
                                        mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> alpha_buf =
        distributed::MeshBuffer::create(scalar_size, acc_dram_config,
                                        mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> beta_buf =
        distributed::MeshBuffer::create(scalar_size, acc_dram_config,
                                        mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> out_buf =
        distributed::MeshBuffer::create(y_size, acc_dram_config,
                                        mesh_device.get());
    // ANCHOR_END: host_buffers

    Program program = CreateProgram();

    // ANCHOR: host_cbs
    // c_0 (A) e c_1 (x): bfloat16, 2 tiles (double buffering).
    // c_2 (y): float, 2 tiles.
    // c_3 (alpha) e c_4 (beta): float, 1 tile, lidos uma vez e nunca liberados.
    // c_5: resultado intermediário A x (float), entre a fase de matmul e a de
    // SFPU. c_16 (saída): float, 2 tiles.
    constexpr uint32_t tiles_per_cb = 2;

    CircularBufferConfig cb_a_config =
        CircularBufferConfig(tiles_per_cb * in_tile_bytes,
                             {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_0, in_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_a_config);

    CircularBufferConfig cb_x_config =
        CircularBufferConfig(tiles_per_cb * in_tile_bytes,
                             {{tt::CBIndex::c_1, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_1, in_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_x_config);

    CircularBufferConfig cb_y_config =
        CircularBufferConfig(tiles_per_cb * acc_tile_bytes,
                             {{tt::CBIndex::c_2, tt::DataFormat::Float32}})
            .set_page_size(tt::CBIndex::c_2, acc_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_y_config);

    CircularBufferConfig cb_alpha_config =
        CircularBufferConfig(acc_tile_bytes,
                             {{tt::CBIndex::c_3, tt::DataFormat::Float32}})
            .set_page_size(tt::CBIndex::c_3, acc_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_alpha_config);

    CircularBufferConfig cb_beta_config =
        CircularBufferConfig(acc_tile_bytes,
                             {{tt::CBIndex::c_4, tt::DataFormat::Float32}})
            .set_page_size(tt::CBIndex::c_4, acc_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_beta_config);

    CircularBufferConfig cb_ax_config =
        CircularBufferConfig(acc_tile_bytes,
                             {{tt::CBIndex::c_5, tt::DataFormat::Float32}})
            .set_page_size(tt::CBIndex::c_5, acc_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_ax_config);

    CircularBufferConfig cb_out_config =
        CircularBufferConfig(tiles_per_cb * acc_tile_bytes,
                             {{tt::CBIndex::c_16, tt::DataFormat::Float32}})
            .set_page_size(tt::CBIndex::c_16, acc_tile_bytes);
    CreateCircularBuffer(program, all_cores, cb_out_config);
    // ANCHOR_END: host_cbs

    std::vector<uint32_t> reader_args;
    TensorAccessorArgs(*a_buf).append_to(reader_args);
    TensorAccessorArgs(*x_buf).append_to(reader_args);
    TensorAccessorArgs(*y_buf).append_to(reader_args);
    TensorAccessorArgs(*alpha_buf).append_to(reader_args);
    TensorAccessorArgs(*beta_buf).append_to(reader_args);
    std::vector<uint32_t> writer_args;
    TensorAccessorArgs(*out_buf).append_to(writer_args);

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
    // fp32_dest_acc_en: o Dst guarda FP32, então a soma dos Nt produtos de
    // tiles não perde precisão. HiFi4 usa todos os bits da mantissa no FPU.
    KernelHandle compute = CreateKernel(
        program, KERNEL_DIR "compute/compute.cpp", all_cores,
        ComputeConfig{.math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
                      .fp32_dest_acc_en = true});
    // ANCHOR_END: host_kernels

    // ANCHOR: host_runtime_args
    const std::vector<std::pair<tt::tt_metal::CoreRangeSet, uint32_t>>
        work_groups = {{group_1, rows_per_core_1}, {group_2, rows_per_core_2}};

    uint32_t start_row = 0;
    for (const std::pair<tt::tt_metal::CoreRangeSet, uint32_t> &work_group :
         work_groups) {
        const uint32_t rows_per_core = work_group.second;
        for (const tt::tt_metal::CoreRange &range : work_group.first.ranges()) {
            for (const CoreCoord &core : range) {
                SetRuntimeArgs(program, reader, core,
                               {static_cast<uint32_t>(a_buf->address()),
                                static_cast<uint32_t>(x_buf->address()),
                                static_cast<uint32_t>(y_buf->address()),
                                static_cast<uint32_t>(alpha_buf->address()),
                                static_cast<uint32_t>(beta_buf->address()), Mt,
                                Nt, start_row, rows_per_core});
                SetRuntimeArgs(program, compute, core, {Nt, rows_per_core});
                SetRuntimeArgs(program, writer, core,
                               {static_cast<uint32_t>(out_buf->address()), Mt,
                                start_row, rows_per_core});
                start_row += rows_per_core;
            }
        }
    }
    // ANCHOR_END: host_runtime_args

    // ANCHOR: host_enqueue
    distributed::EnqueueWriteMeshBuffer(cq, a_buf, a_tiled, /*blocking=*/false);
    distributed::EnqueueWriteMeshBuffer(cq, x_buf, x_tiled, /*blocking=*/false);
    distributed::EnqueueWriteMeshBuffer(cq, y_buf, y_tiled, /*blocking=*/false);
    distributed::EnqueueWriteMeshBuffer(cq, alpha_buf, alpha_tiled,
                                        /*blocking=*/false);
    distributed::EnqueueWriteMeshBuffer(cq, beta_buf, beta_tiled,
                                        /*blocking=*/false);

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()),
                         std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

    std::vector<float> out_tiled;
    distributed::EnqueueReadMeshBuffer(cq, out_tiled, out_buf,
                                       /*blocking=*/true);
    // ANCHOR_END: host_enqueue

    // ANCHOR: host_unpad
    // O resultado tem o mesmo formato de y_padded: y[i] está na coluna 0.
    const std::vector<float> out =
        convert_layout<float>(out_tiled, {m, m}, TensorLayoutType::TILED_NFACES,
                              TensorLayoutType::LIN_ROW_MAJOR);
    std::vector<float> y_out(m);
    for (uint32_t i = 0; i < m; i++) {
        y_out[i] = out[i * m];
    }
    // ANCHOR_END: host_unpad

    const sscad::CompareResult result = sscad::compare(
        y_out, sscad::reference::gemv(m, n, alpha, a, x, beta, y));
    sscad::print_compare(result);

    const bool closed = mesh_device->close();
    return sscad::report(closed && result.pass);
}
