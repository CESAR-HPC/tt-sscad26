#!/usr/bin/env bash
# Configura e compila todos os exemplos em build/.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

cmake -G Ninja -B build -S . "$@"
cmake --build build
