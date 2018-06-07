cmake_minimum_required (VERSION 3.1)
project (bcnn2tflite)

set(CMAKE_CXX_STANDARD 11)

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
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/../../inc
    ${PROJECT_SOURCE_DIR}/../../bh/inc
    )

add_executable(bcnn2tflite bcnn2tflite.cc)

if(NOT MSVC)
    if (USE_CUDA) 
        target_link_libraries(bcnn2tflite bcnn bip -lstdc++ -lm)
    else()
        target_link_libraries(bcnn2tflite bcnn bip -lm)
    endif()
else()
    target_link_libraries(bcnn2tflite bcnn bip)
endif()
