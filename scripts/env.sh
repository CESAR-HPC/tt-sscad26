# Variáveis de ambiente do TT-Metalium + simulador. Uso: source scripts/env.sh
#
# Valores já definidos no ambiente (por exemplo, pelo docker-compose ou por uma
# instalação local) são preservados.
export TT_METAL_HOME="${TT_METAL_HOME:-/tt-metal}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
export ARCH_NAME="${ARCH_NAME:-wormhole_b0}"
export TT_METAL_SIMULATOR="${TT_METAL_SIMULATOR:-/sim/libttsim_wh.so}"
# O simulador só suporta o modo de despacho lento.
export TT_METAL_SLOW_DISPATCH_MODE=1
