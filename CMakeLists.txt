cmake_minimum_required (VERSION 2.9)
project (bcnn)

set(USE_SSE2 1)
set(USE_CUDA 1)
set(BUILD_EXAMPLES 1)

if (USE_CUDA)
find_package(CUDA QUIET REQUIRED)
add_definitions(-DBCNN_USE_CUDA)
set(CUDA_NVCC_FLAGS "-arch=compute_20; -code=sm_20; -lcuda -lcudart -lcublas -lcurand")
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
	${PROJECT_SOURCE_DIR}/inc
	${PROJECT_SOURCE_DIR}/bip/inc
	${PROJECT_SOURCE_DIR}/bh/inc
	${PROJECT_SOURCE_DIR}/bip/thirdparty
	)

add_subdirectory(bip) 

if (USE_CUDA)
file(GLOB SRC_LIB src/*.c src/*.cu)
else()
file(GLOB SRC_LIB src/*.c)
endif()
list(REMOVE_ITEM SRC_LIB ${CMAKE_CURRENT_SOURCE_DIR}/src/bcnn_cl.c)
file(GLOB SRC_CLI src/bcnn_cl.c)
MESSAGE(STATUS "${SRC_LIB}")
MESSAGE(STATUS "${CMAKE_CURRENT_SOURCE_DIR}")

if (USE_CUDA)
cuda_add_library(libbcnn ${SRC_LIB} STATIC)
target_link_libraries(libbcnn bip ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY})
else()
add_library(libbcnn STATIC ${SRC_LIB})
target_link_libraries(libbcnn bip)
endif()

add_executable(bcnn-cl ${SRC_CLI})
target_link_libraries(bcnn-cl libbcnn bip)

add_subdirectory(examples/mnist) 