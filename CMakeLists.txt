cmake_minimum_required (VERSION 2.9)
project (bcnn)

# User configuration settings
option(USE_AVX "Build with AVX instructions" ON)
option(USE_CUDA "Build with CUDA libraries" OFF)
option(USE_CUDNN "Build with CuDNN library" OFF)
option(USE_BLAS "Build with BLAS library" ON)
option(USE_NEON "Build with Neon instructions" OFF)
# Building examples
option(BUILD_EXAMPLES "Build examples" ON)
# Building tools
option(BUILD_TOOLS "Build tools" OFF)
# Set OFF to turn off logging
set(ENABLE_LOG ON)

# Convenience stuff
include(CMakeToolsHelpers OPTIONAL)

# Build directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

enable_language(C ASM)
# C99
set (CMAKE_C_STANDARD 99)

# Compiler flags
set(CMAKE_C_FLAGS "-O3")
set(CMAKE_ASM_FLAGS "${CFLAGS} -x assembler-with-cpp")

# Get the target system processor
message(STATUS "[bcnn] Targeted system processor: " ${CMAKE_SYSTEM_PROCESSOR})

# Blas backend
if (USE_BLAS)
    add_definitions(-DBCNN_USE_BLAS)
    # Try to find openblas first
    set(OpenBLAS_INC_PATHS /opt/OpenBLAS/include /usr/include /usr/include/openblas
        /usr/local/include
        /usr/local/include/openblas
    )
    set(OpenBLAS_LIB_PATHS /opt/OpenBLAS/lib /opt/OpenBLAS/lib64 /lib/ /lib64/ /usr/lib /usr/lib64
        /usr/local/lib
        /usr/local/lib64
    )
    find_path(OpenBLAS_INCLUDE NAMES cblas.h PATHS ${OpenBLAS_INC_PATHS})
    find_library(OpenBLAS_LIB NAMES openblas PATHS ${OpenBLAS_LIB_PATHS})
    if(OpenBLAS_INCLUDE AND OpenBLAS_LIB)
        set(OpenBLAS_FOUND ON)
        message(STATUS "[bcnn] Found OpenBlas here: " ${OpenBLAS_LIB})
	include_directories(${OpenBLAS_INC_PATHS})
        set(BLAS_LIBRARY ${OpenBLAS_LIB})
    endif()
    # If OpenBlas is not found, try to find cblas
    if(NOT OpenBLAS_FOUND)
        find_package(BLAS QUIET)
        find_library(CBLAS_LIB NAMES cblas PATHS)
        if (BLAS_FOUND AND CBLAS_LIB)
            message(STATUS "[bcnn] Found Blas here: " ${CBLAS_LIB})
            set(BLAS_LIBRARY ${CBLAS_LIB})
        elseif(NOT CBLAS_FOUND)
            message(FATAL_ERROR "[bcnn] Could not find any Blas library")
        endif()
    endif()
endif()

# CUDA backend
if (USE_CUDA)
    find_package(CUDA QUIET REQUIRED)
    add_definitions(-DBCNN_USE_CUDA)
    message(STATUS "[bcnn] Build with Cuda")
    message(STATUS "[bcnn] Found Cuda libraries: " ${CUDA_CUBLAS_LIBRARIES})
    # Uncomment the proper line according to the system cuda arch
    set(CUDA_ARCH 
        #"-gencode arch=compute_30,code=sm_30;"
        #"-gencode arch=compute_35,code=sm_35;"
        "-gencode arch=compute_50,code=sm_50;"
        #"-gencode arch=compute_50,code=compute_50;"
        #"-gencode arch=compute_52,code=sm_52;"
        #"-gencode arch=compute_60,code=sm_60;"
        #"-gencode arch=compute_61,code=sm_61;"
    )
    set(CUDA_NVCC_FLAGS ${CUDA_ARCH} "-lcuda -lcudart -lcublas -lcurand")
    #list(APPEND CMAKE_CXX_SOURCE_FILE_EXTENSIONS c)
    if (USE_CUDNN)
        message(STATUS "[bcnn] Build with CuDNN")
        set(CUDNN_ROOT_DIR "" CACHE PATH "Folder contains NVIDIA cuDNN")
        find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES cuda/include include)
        find_library(CUDNN_LIBRARY cudnn HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)
        set(CUDNN_INCLUDE_DIRS ${CUDNN_INCLUDE_DIR})
        set(CUDNN_LIBRARIES ${CUDNN_LIBRARY})
        add_definitions(-DBCNN_USE_CUDNN)
    endif()
endif()

if (USE_AVX)
    message(STATUS "[bcnn] Build with AVX instructions")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2")
    add_definitions(-DBCNN_USE_AVX)
endif()
if (USE_NEON)
    message(STATUS "[bcnn] Build with Neon instructions")
    # Note: Uncomment below line in case of older arm architecture
    #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfpu=neon")
    add_definitions(-DBCNN_USE_NEON)
endif()

if (ENABLE_LOG)
    message(STATUS "[bcnn] Enabling logging")
    add_definitions(-DBCNN_LOG_ENABLED)
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
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/inc
    ${PROJECT_SOURCE_DIR}/bip/inc
    ${PROJECT_SOURCE_DIR}/bh/inc
    ${PROJECT_SOURCE_DIR}/bip/thirdparty
    )

add_subdirectory(bip) 

if (USE_CUDA)
file(GLOB SRC_LIB src/*.c src/*.cu)
else()
    if(USE_NEON AND CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        file(GLOB SRC_LIB src/*.c src/openblas/*.c src/openblas/generic/*.c src/openblas/arm64/*.S)
    else()
        file(GLOB SRC_LIB src/*.c)
    endif()
endif()
list(REMOVE_ITEM SRC_LIB ${CMAKE_CURRENT_SOURCE_DIR}/src/bcnn_cl.c)
file(GLOB SRC_CLI src/bcnn_cl.c)
#MESSAGE(STATUS "${SRC_LIB}")
#MESSAGE(STATUS "${CMAKE_CURRENT_SOURCE_DIR}")

if (USE_CUDA)
    cuda_add_library(bcnn ${SRC_LIB} STATIC)
    if (USE_CUDNN)
        target_link_libraries(bcnn bip ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY}
            ${CUDNN_LIBRARIES})
    else()
        target_link_libraries(bcnn bip ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY})
    endif()
else()
    add_library(bcnn STATIC ${SRC_LIB})
    target_link_libraries(bcnn bip ${BLAS_LIBRARY})
endif()

add_executable(bcnn-cl ${SRC_CLI})

if(NOT MSVC)
    if (USE_CUDA) 
        target_link_libraries(bcnn-cl bcnn bip -lstdc++ -lm)
    else()
        target_link_libraries(bcnn-cl bcnn bip -lm)
    endif()
else()
    target_link_libraries(bcnn-cl bcnn bip)
endif()


if (BUILD_EXAMPLES)
add_subdirectory(examples/mnist)
add_subdirectory(examples/cifar10)
add_subdirectory(examples/yolo)
endif()

if (BUILD_TOOLS)
    #add_subdirectory(tools/caffe_converter)
    add_subdirectory(tools/tflite_converter)
endif()

