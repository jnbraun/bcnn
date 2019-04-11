cmake_minimum_required (VERSION 3.0)
project (inference-bench)

include_directories (
    ${PROJECT_SOURCE_DIR}/../../inc
    ${PROJECT_SOURCE_DIR}/../../bh/inc
    )

file(GLOB SRC *.c)
add_executable(inference-bench ${SRC})

target_link_libraries(inference-bench bcnn -lstdc++ -lm)