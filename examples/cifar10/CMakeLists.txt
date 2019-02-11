cmake_minimum_required (VERSION 2.9)
project (cifar10-example)

if(MSVC)
    set(variables
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_MINSIZEREL
        CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_MINSIZEREL
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_RELWITHDEBINFO
    )
    foreach(variable ${variables})
        if(${variable} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MT" ${variable} "${${variable}}")
        endif()
    endforeach()
endif()

include_directories (
    ${PROJECT_SOURCE_DIR}/../../inc
    ${PROJECT_SOURCE_DIR}/../../bh/inc
    )

file(GLOB SRC *.c)
add_executable(cifar10-example ${SRC})
#target_link_libraries(cifar10-example bcnn)

if(NOT MSVC)
    if (USE_CUDA) 
        target_link_libraries(cifar10-example bcnn -lstdc++ -lm)
    else()
        target_link_libraries(cifar10-example bcnn -lm)
    endif()
else()
    target_link_libraries(cifar10-example bcnn)
endif()
