cmake_minimum_required (VERSION 2.9)
project (bcnn)

# User configuration settings
option(USE_AVX "Build with AVX instructions" ON)
option(USE_CUDA "Build with CUDA libraries" OFF)
option(USE_CUDNN "Build with CuDNN library" OFF)
option(USE_BLAS "Build with BLAS library" OFF)
option(USE_NEON "Build with Neon instructions" OFF)
option(USE_VULKAN "Build with Vulkan support" OFF)
option(USE_OPENMP "Enable OpenMP multithreading" ON)
# Building examples
option(BUILD_EXAMPLES "Build examples" ON)
# Building tools
option(BUILD_TOOLS "Build tools" ON)
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
set(CMAKE_C_FLAGS "-O3 -fPIC")
set(CMAKE_ASM_FLAGS "${CFLAGS} -x assembler-with-cpp")

# Get the target system processor
message(STATUS "[bcnn] Targeted system processor: " ${CMAKE_SYSTEM_PROCESSOR})

# Blas backend
if (USE_BLAS)
    # Check that USE_CUDA is turn off when using Blas
    if (USE_CUDA OR USE_CUDNN)
        message(FATAL_ERROR 
        "[bcnn] Can not use Blas when USE_CUDA is turned ON.
        Please turn off USE_CUDA option if you want to run on CPU or
        turn off USE_BLAS if you want to use CUDA")
    else()
        add_definitions(-DBCNN_USE_BLAS)
    endif()
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
        message(STATUS "[bcnn] Build with OpenBlas found here: " ${OpenBLAS_LIB})
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
    if (USE_BLAS)
        add_definitions(-DBCNN_USE_BLAS=OFF)
    endif()
    message(STATUS "[bcnn] Build with Cuda")
    message(STATUS "[bcnn] Found Cuda libraries: " ${CUDA_CUBLAS_LIBRARIES})
    # Uncomment the proper line according to the system cuda arch
    set(CUDA_ARCH 
        #"-gencode arch=compute_30,code=sm_30;"
        #"-gencode arch=compute_35,code=sm_35;"
        "-gencode arch=compute_50,code=sm_50;"
        "-gencode arch=compute_50,code=compute_50;"
        "-gencode arch=compute_52,code=sm_52;"
        #"-gencode arch=compute_60,code=sm_60;"
        #"-gencode arch=compute_61,code=sm_61;"
    )
    set(CUDA_NVCC_FLAGS ${CUDA_ARCH} "-lcuda -lcudart -lcublas -lcurand")
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

# Vulkan backend
if (USE_VULKAN)
    if (USE_CUDA OR USE_CUDNN)
        message(FATAL_ERROR "[bcnn] Can not use Vulkan when USE_CUDA is turned ON.")
    else()
        if (NOT CMAKE_VERSION VERSION_LESS 3.7.0)
            find_package(Vulkan)
        endif()
        if (NOT Vulkan_FOUND)
            find_library(Vulkan_LIBRARY NAMES vulkan HINTS "$ENV{VULKAN_SDK}/lib" "${CMAKE_SOURCE_DIR}/libs/vulkan" REQUIRED)
            if (Vulkan_LIBRARY)
                set(Vulkan_FOUND ON)
                message(STATUS "[bcnn] Build with Vulkan found here: " ${Vulkan_LIBRARY} " " ${Vulkan_INCLUDE_DIR})
                include_directories(${Vulkan_INCLUDE_DIR})
                add_definitions(-DBCNN_USE_VULKAN)
            endif()
        endif()
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

# OpenMP
if (USE_OPENMP)
    find_package(OpenMP REQUIRED)
    if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
        message(STATUS "[bcnn] Found openmp")
        add_definitions(-DBCNN_USE_OPENMP)
    endif()
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
    ${PROJECT_SOURCE_DIR}/src/layers
    ${PROJECT_SOURCE_DIR}/src/kernels
    ${PROJECT_SOURCE_DIR}/inc
    ${PROJECT_SOURCE_DIR}/src/bip/inc
    ${PROJECT_SOURCE_DIR}/src/bh/inc
    ${PROJECT_SOURCE_DIR}/src/bip/thirdparty
    )

add_subdirectory(src/bip) 

if (USE_CUDA)
    file(GLOB SRC_LIB src/*.c src/*.cu src/kernels/*.c src/kernels/*.cu
            src/layers/*.c src/layers/*.cu src/data_loader/*.c)
else()
    if(USE_NEON AND CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        file(GLOB SRC_LIB src/*.c src/layers/*.c src/kernels/*.c src/data_loader/*.c
            src/kernels/openblas/*.c src/kernels/openblas/generic/*.c
            src/kernels/openblas/arm64/*.S)
    else()
        file(GLOB SRC_LIB src/*.c src/kernels/*.c src/layers/*.c src/data_loader/*.c)
    endif()
endif()
file(GLOB SRC_CLI src/cli/*.c)

if (USE_CUDA)
    cuda_add_library(bcnn ${SRC_LIB} STATIC)
    if (USE_CUDNN)
        target_link_libraries(bcnn bip ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY}
            ${CUDNN_LIBRARIES})
    else()
        target_link_libraries(bcnn bip ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY})
    endif()
else()
    if(BCNN_USE_VULKAN)
        # TODO add shader sources
        add_library(bcnn STATIC ${SRC_LIB})
        add_dependencies(bcnn generate-spirv)
        target_link_libraries(bcnn bip ${Vulkan_LIBRARY})
    else()
        add_library(bcnn STATIC ${SRC_LIB})
        target_link_libraries(bcnn bip ${BLAS_LIBRARY})
    endif()
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
    add_subdirectory(examples/inference_benchmark)
endif()

if (BUILD_TOOLS)
    #add_subdirectory(tools/caffe_converter)
    add_subdirectory(tools/tflite_converter)
    #add_subdirectory(tools/tensorflow_converter)
endif()

