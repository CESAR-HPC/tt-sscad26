# Utilitários de host, apenas headers (namespace sscad).
add_library(sscad_common INTERFACE)
target_include_directories(sscad_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sscad_common INTERFACE sscad_tt_metal)
