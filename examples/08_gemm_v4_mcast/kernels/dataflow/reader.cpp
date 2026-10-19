// Reader do exemplo 08 (GEMM v4). Um kernel só para todos os núcleos: o
// papel do núcleo sai da posição do seu bloco (by, bx) na grade.
//   - bx == 0: emissor de A. Lê o bloco de A da DRAM (como no exemplo 07) e o
//     envia por multicast aos núcleos (1..X-1, by). Senão, recebe A do núcleo
//     (0, by).
//   - by == 0: emissor de B. Lê o bloco de B e o envia aos núcleos
//     (bx, 1..Y-1). Senão, recebe B do núcleo (bx, 0).
//
// Protocolo para cada bloco (igual para A e B):
//   receptor: reserva espaço no CB, marca o próprio *_valid como INVALID,
//             incrementa o *_ready do emissor e espera *_valid == VALID;
//   emissor:  lê o bloco, espera *_ready == número de receptores (todos têm
//             espaço), zera *_ready, envia o bloco por multicast e depois o
//             valor VALID do seu *_valid para o *_valid de cada receptor.
// O bloco é escrito no mesmo endereço do CB em todos os núcleos: os CBs têm a
// mesma configuração e avançam em sincronia, então o ponteiro de escrita do
// emissor é também o dos receptores.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t Kt = get_arg_val<uint32_t>(2);
    const uint32_t Nt = get_arg_val<uint32_t>(3);
    const uint32_t block_y = get_arg_val<uint32_t>(4);
    const uint32_t block_x = get_arg_val<uint32_t>(5);
    const uint32_t per_core_M = get_arg_val<uint32_t>(6);
    const uint32_t per_core_N = get_arg_val<uint32_t>(7);
    const uint32_t in0_block_w = get_arg_val<uint32_t>(8);
    // Multicast de A: coordenadas NoC do emissor e do retângulo de receptores.
    const uint32_t a_sender_x = get_arg_val<uint32_t>(9);
    const uint32_t a_sender_y = get_arg_val<uint32_t>(10);
    const uint32_t a_mcast_start_x = get_arg_val<uint32_t>(11);
    const uint32_t a_mcast_start_y = get_arg_val<uint32_t>(12);
    const uint32_t a_mcast_end_x = get_arg_val<uint32_t>(13);
    const uint32_t a_mcast_end_y = get_arg_val<uint32_t>(14);
    const uint32_t a_num_receivers = get_arg_val<uint32_t>(15);
    // Multicast de B.
    const uint32_t b_sender_x = get_arg_val<uint32_t>(16);
    const uint32_t b_sender_y = get_arg_val<uint32_t>(17);
    const uint32_t b_mcast_start_x = get_arg_val<uint32_t>(18);
    const uint32_t b_mcast_start_y = get_arg_val<uint32_t>(19);
    const uint32_t b_mcast_end_x = get_arg_val<uint32_t>(20);
    const uint32_t b_mcast_end_y = get_arg_val<uint32_t>(21);
    const uint32_t b_num_receivers = get_arg_val<uint32_t>(22);
    // ids dos semáforos (CreateSemaphore no host).
    const uint32_t a_ready_id = get_arg_val<uint32_t>(23);
    const uint32_t a_valid_id = get_arg_val<uint32_t>(24);
    const uint32_t b_ready_id = get_arg_val<uint32_t>(25);
    const uint32_t b_valid_id = get_arg_val<uint32_t>(26);

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;

    constexpr TensorAccessorArgs<0> a_args;
    const TensorAccessor a(a_args, a_addr);
    constexpr TensorAccessorArgs<a_args.next_compile_time_args_offset()> b_args;
    const TensorAccessor b(b_args, b_addr);

    Noc noc;
    CircularBuffer a_cb(cb_a);
    CircularBuffer b_cb(cb_b);
    const uint32_t a_tile_bytes = a_cb.get_tile_size();
    const uint32_t b_tile_bytes = b_cb.get_tile_size();

    // ANCHOR: reader_semaphores
    Semaphore a_ready(a_ready_id);
    Semaphore a_valid(a_valid_id);
    Semaphore b_ready(b_ready_id);
    Semaphore b_valid(b_valid_id);

    const bool a_sender = block_x == 0;
    const bool b_sender = block_y == 0;
    // O emissor envia aos receptores o valor do seu próprio *_valid, que fica
    // sempre VALID.
    if (a_sender) {
        a_valid.set(VALID);
    }
    if (b_sender) {
        b_valid.set(VALID);
    }
    // ANCHOR_END: reader_semaphores

    const uint32_t a_block_tiles = per_core_M * in0_block_w;
    const uint32_t b_block_tiles = in0_block_w * per_core_N;
    const uint32_t num_k_blocks = Kt / in0_block_w;
    const uint32_t a_row0 = block_y * per_core_M;
    const uint32_t b_col0 = block_x * per_core_N;

    // ANCHOR: reader_loop
    for (uint32_t kb = 0; kb < num_k_blocks; kb++) {
        const uint32_t k0 = kb * in0_block_w;

        // Bloco de A (per_core_M x in0_block_w), da coluna 0 para a linha.
        a_cb.reserve_back(a_block_tiles);
        if (a_sender) {
            uint32_t slot = 0;
            for (uint32_t h = 0; h < per_core_M; h++) {
                for (uint32_t w = 0; w < in0_block_w; w++) {
                    const uint32_t tile = (a_row0 + h) * Kt + (k0 + w);
                    noc.async_read(a, a_cb, a_tile_bytes, {.page_id = tile},
                                   {.offset_bytes = slot * a_tile_bytes});
                    slot++;
                }
            }
            noc.async_read_barrier();

            // ANCHOR: reader_mcast
            // Espera todos os receptores terem espaço no CB e rearma o contador.
            a_ready.wait(a_num_receivers);
            a_ready.set(0);

            // O bloco sai do ponteiro de escrita do CB local (onde acabou de
            // ser lido) e vai para o mesmo endereço em cada receptor.
            noc.async_write_multicast(
                use<CircularBuffer::AddrSelector::WRITE_PTR>(a_cb), a_cb,
                a_block_tiles * a_tile_bytes, a_num_receivers,
                {.offset_bytes = 0},
                {.noc_x_start = a_mcast_start_x,
                 .noc_y_start = a_mcast_start_y,
                 .noc_x_end = a_mcast_end_x,
                 .noc_y_end = a_mcast_end_y,
                 .offset_bytes = 0});
#ifdef ARCH_BLACKHOLE
            // No Blackhole as duas escritas podem sair fora de ordem.
            noc.async_writes_flushed();
#endif
            // Na mesma NoC, o semáforo chega depois dos dados.
            a_valid.set_multicast(noc, a_mcast_start_x, a_mcast_start_y,
                                  a_mcast_end_x, a_mcast_end_y,
                                  a_num_receivers);
            // ANCHOR_END: reader_mcast
        } else {
            // ANCHOR: reader_receive
            a_valid.set(INVALID);
            a_ready.up(noc, a_sender_x, a_sender_y, 1);
            a_valid.wait(VALID);
            // ANCHOR_END: reader_receive
        }
        a_cb.push_back(a_block_tiles);

        // Bloco de B (in0_block_w x per_core_N), da linha 0 para a coluna.
        b_cb.reserve_back(b_block_tiles);
        if (b_sender) {
            uint32_t slot = 0;
            for (uint32_t h = 0; h < in0_block_w; h++) {
                for (uint32_t w = 0; w < per_core_N; w++) {
                    const uint32_t tile = (k0 + h) * Nt + (b_col0 + w);
                    noc.async_read(b, b_cb, b_tile_bytes, {.page_id = tile},
                                   {.offset_bytes = slot * b_tile_bytes});
                    slot++;
                }
            }
            noc.async_read_barrier();

            b_ready.wait(b_num_receivers);
            b_ready.set(0);

            noc.async_write_multicast(
                use<CircularBuffer::AddrSelector::WRITE_PTR>(b_cb), b_cb,
                b_block_tiles * b_tile_bytes, b_num_receivers,
                {.offset_bytes = 0},
                {.noc_x_start = b_mcast_start_x,
                 .noc_y_start = b_mcast_start_y,
                 .noc_x_end = b_mcast_end_x,
                 .noc_y_end = b_mcast_end_y,
                 .offset_bytes = 0});
#ifdef ARCH_BLACKHOLE
            noc.async_writes_flushed();
#endif
            b_valid.set_multicast(noc, b_mcast_start_x, b_mcast_start_y,
                                  b_mcast_end_x, b_mcast_end_y,
                                  b_num_receivers);
        } else {
            b_valid.set(INVALID);
            b_ready.up(noc, b_sender_x, b_sender_y, 1);
            b_valid.wait(VALID);
        }
        b_cb.push_back(b_block_tiles);
    }
    // ANCHOR_END: reader_loop

    // Garante que os multicasts terminaram antes de o kernel acabar.
    noc.async_write_barrier();
}
