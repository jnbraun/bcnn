cmake_minimum_required (VERSION 2.9)
project (yolov2-example)

if (USE_CUDA)
add_definitions(-DBCNN_USE_CUDA)
endif()

if (USE_SSE2)
add_definitions(-DBCNN_USE_SSE2)
endif()

if (USE_CUDNN)
add_definitions(-DBCNN_USE_CUDNN)
endif()

add_definitions(-DBIP_USE_STB_IMAGE)

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

file(GLOB SRC *.cc *.c)
add_executable(yolov2-example ${SRC})

find_package(OpenCV 3.0.0 REQUIRED)
if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    if(NOT MSVC)
        if (USE_CUDA) 
            target_link_libraries(yolov2-example ${OpenCV_LIBS} bcnn bip -lstdc++ -lm)
        else()
            target_link_libraries(yolov2-example ${OpenCV_LIBS} bcnn bip -lm)
        endif()
    else()
        target_link_libraries(yolov2-example ${OpenCV_LIBS} bcnn bip)
    endif()
else()
   message(FATAL_ERROR, "Could not find OpenCV")
endif()
