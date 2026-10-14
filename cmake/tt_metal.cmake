# Alvo INTERFACE `sscad_tt_metal`: tudo o que um exemplo precisa para usar o
# TT-Metalium (includes, bibliotecas e flags).
#
# Usamos o pacote CMake que o próprio tt-metal instala em
# $TT_METAL_HOME/build_Release (lib/cmake/tt-metalium). Os diretórios de
# include de alvos IMPORTED entram como SYSTEM, então avisos dos headers do
# tt-metal não aparecem, enquanto o nosso código compila com
# -Werror=deprecated-declarations.

if("$ENV{TT_METAL_HOME}" STREQUAL "")
    message(FATAL_ERROR "TT_METAL_HOME não está definido (rode `source scripts/env.sh`)")
endif()

# Aponta para a configuração instalada (lib/cmake/...), e não para a cópia
# que fica na raiz de build_Release e não encontra Metalium.cmake.
set(TT_METAL_PREFIX "$ENV{TT_METAL_HOME}/build_Release")
list(PREPEND CMAKE_PREFIX_PATH "${TT_METAL_PREFIX}")
set(TT-Metalium_DIR "${TT_METAL_PREFIX}/lib/cmake/tt-metalium" CACHE PATH "")
find_package(TT-Metalium CONFIG REQUIRED)

add_library(sscad_tt_metal INTERFACE)
target_link_libraries(sscad_tt_metal INTERFACE TT::Metalium)
target_compile_features(sscad_tt_metal INTERFACE cxx_std_20)
target_compile_options(sscad_tt_metal INTERFACE -Wall -Wextra -Werror=deprecated-declarations
    # Os exemplos usam inicializadores designados parciais (idioma do
    # TT-Metalium para DataMovementConfig etc.); os campos omitidos têm padrão.
    -Wno-missing-field-initializers)

# sscad_add_example(<executável> <fontes...>)
#
# Cria o executável em build/bin/, liga-o ao TT-Metalium e a common/, e passa
# o diretório de kernels do exemplo como KERNEL_DIR (caminho absoluto), usado
# em CreateKernel(program, KERNEL_DIR "dataflow/reader.cpp", ...).
function(sscad_add_example name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE sscad_common)
    target_compile_definitions(${name} PRIVATE KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels/")
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endfunction()
