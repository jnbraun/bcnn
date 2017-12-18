cmake_minimum_required (VERSION 2.9)
project (caffe2bcnn)

# Find caffe
# If caffe could not be found then set manually the paths
#set(Caffe_DIR "/path/to/caffe")
#set(Caffe_INCLUDE_DIRS "/path/to/caffe/include")
#set(Caffe_LIBRARIES "/path/to/caffe/lib/libcaffeproto.a" ...)
find_package(Caffe REQUIRED)
include_directories(${Caffe_INCLUDE_DIRS})

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

add_executable(caffe2bcnn convert.cpp)

if(NOT MSVC)
    if (USE_CUDA) 
        target_link_libraries(caffe2bcnn libbcnn bip -lstdc++ -lm ${Caffe_LIBRARIES} -lboost_system -lglog)
    else()
        target_link_libraries(caffe2bcnn libbcnn bip -lm ${Caffe_LIBRARIES} -lboost_system -lglog)
    endif()
else()
    target_link_libraries(caffe2bcnn libbcnn bip ${Caffe_LIBRARIES} -lboost_system -lglog)
endif()
