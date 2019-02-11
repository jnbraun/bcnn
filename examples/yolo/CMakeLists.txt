cmake_minimum_required (VERSION 2.9)
project (yolo-example)

option(USE_OPENCV "Use OpenCV for display" ON)

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
add_executable(yolo-example ${SRC})

find_package(OpenCV 3.0.0)
if (OpenCV_FOUND AND USE_OPENCV)
    message(STATUS "[bcnn] [examples/yolo] Found OpenCV")
    add_definitions(-DUSE_OPENCV)
    include_directories(${OpenCV_INCLUDE_DIRS})
    if(NOT MSVC)
        if (USE_CUDA) 
            target_link_libraries(yolo-example ${OpenCV_LIBS} bcnn bip -lstdc++ -lm)
        else()
            target_link_libraries(yolo-example ${OpenCV_LIBS} bcnn bip -lm)
        endif()
    else()
        target_link_libraries(yolo-example ${OpenCV_LIBS} bcnn bip)
    endif()
else()
   message(STATUS "[bcnn] [examples/yolo] Could not find OpenCV: won't be able to run with the webcam")
   if(NOT MSVC)
        if (USE_CUDA) 
            target_link_libraries(yolo-example bcnn bip -lstdc++ -lm)
        else()
            target_link_libraries(yolo-example bcnn bip -lm)
        endif()
    else()
        target_link_libraries(yolo-example bcnn bip)
    endif()
endif()
