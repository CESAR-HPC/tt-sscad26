#!/usr/bin/env bash
# Executa um exemplo com o ambiente correto.
# Uso: scripts/run.sh <exemplo> [args...]   ex.: scripts/run.sh 02_eltwise_add 128 128
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

if [[ $# -lt 1 ]]; then
    echo "Uso: $0 <exemplo> [args...]" >&2
    exit 2
fi
example="${1%/}"
example="${example#examples/}"
shift
exec "build/bin/ex${example}" "$@"
