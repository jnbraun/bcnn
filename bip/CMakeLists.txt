cmake_minimum_required (VERSION 2.9)
project (bip)

set (USE_STB_IMAGE, 1)

if(MSVC)
    set(variables
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_MINSIZEREL
        CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_RELWITHDEBINFO
    )
    foreach(variable ${variables})
        if(${variable} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MT" ${variable} "${${variable}}")
    endif()
    endforeach()
endif()

include_directories (
    ${PROJECT_SOURCE_DIR}/inc
    ${PROJECT_SOURCE_DIR}/../bh/inc
    )

if (USE_STB_IMAGE)
add_definitions(-DBIP_USE_STB_IMAGE)
include_directories (${PROJECT_SOURCE_DIR}/thirdparty)
endif()

file(GLOB SOURCES src/bip.c)

add_library(bip STATIC ${SOURCES})