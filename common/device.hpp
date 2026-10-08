// SPDX-License-Identifier: Apache-2.0
//
// Abertura e fechamento do dispositivo Tenstorrent usados por todos os exemplos.
#pragma once

#include <cstdint>
#include <memory>

#include <fmt/core.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

namespace sscad {

namespace distributed = tt::tt_metal::distributed;

// Abre um MeshDevice 1x1, isto é, um único chip. A mesma API escala para
// malhas com vários chips, mas no minicurso trabalhamos sempre com um só.
inline std::shared_ptr<distributed::MeshDevice> open_device(int device_id = 0) {
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    const tt::tt_metal::CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    fmt::print("Dispositivo {} aberto: grade de núcleos Tensix {}x{} ({} núcleos)\n",
               device_id, grid.x, grid.y, grid.x * grid.y);
    return mesh_device;
}

// Fecha o dispositivo; retorna false se o runtime reportar erro.
inline bool close_device(const std::shared_ptr<distributed::MeshDevice> &mesh_device) {
    return mesh_device->close();
}

} // namespace sscad
