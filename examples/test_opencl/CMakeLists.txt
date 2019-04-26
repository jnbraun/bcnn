cmake_minimum_required (VERSION 3.0)
project (test-opencl)

include_directories (
    ${PROJECT_SOURCE_DIR}/../../inc
    ${PROJECT_SOURCE_DIR}/../../bh/inc
    )

file(GLOB SRC *.c)
add_executable(test-opencl ${SRC})

target_link_libraries(test-opencl bcnn -lstdc++ -lm)