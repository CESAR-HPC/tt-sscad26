#!/usr/bin/env bash
# Executa um exemplo com o ambiente correto.
# Uso: scripts/run.sh <exemplo>   ex.: scripts/run.sh 02_eltwise_add
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

if [[ $# -ne 1 ]]; then
    echo "Uso: $0 <exemplo>" >&2
    exit 2
fi
example="${1%/}"
example="${example#examples/}"
exec "build/bin/ex${example}"
