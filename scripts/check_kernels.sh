#!/usr/bin/env bash
# Os kernels são compilados em tempo de execução (JIT), então os avisos de
# API obsoleta não aparecem na compilação do host. Este script procura nos
# kernels as APIs proibidas pelo SPEC e falha se encontrar alguma.
set -euo pipefail
cd "$(dirname "$0")/.."

forbidden=(
    'InterleavedAddrGenFast'
    'InterleavedAddrGen<'
    'noc_async_read_tile'
    'noc_async_write_tile'
    'add_tiles_init'
    'mul_tiles_init'
    'sub_tiles_init'
)

status=0
for symbol in "${forbidden[@]}"; do
    if grep -rnF -- "$symbol" examples/*/kernels; then
        echo "ERRO: API proibida encontrada: $symbol" >&2
        status=1
    fi
done

if [[ $status -eq 0 ]]; then
    echo "check_kernels: nenhuma API proibida nos kernels"
fi
exit $status
