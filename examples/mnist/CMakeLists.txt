cmake_minimum_required (VERSION 2.9)
project (mnist-example)

set(USE_SSE2 1)
set(USE_CUDA 1)

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

file(GLOB SRC *.c)
add_executable(mnist-example ${SRC})
target_link_libraries(mnist-example libbcnn)