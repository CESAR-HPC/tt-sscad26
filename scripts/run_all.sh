#!/usr/bin/env bash
# Executa todos os exemplos implementados (os que não têm o arquivo STUB).
# Sai com código diferente de zero se qualquer um falhar.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

scripts/check_kernels.sh || exit 1

failed=()
for dir in examples/*/; do
    example="$(basename "$dir")"
    if [[ -f "$dir/STUB" ]]; then
        echo "== $example: stub, ignorado"
        continue
    fi
    echo "== $example"
    if ! "build/bin/ex${example}"; then
        failed+=("$example")
    fi
done

if [[ ${#failed[@]} -gt 0 ]]; then
    echo "FALHARAM: ${failed[*]}" >&2
    exit 1
fi
echo "Todos os exemplos implementados passaram."
