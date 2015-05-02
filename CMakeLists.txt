# Object Cadenza
#
# Copyright (c) 2014 Samsung Electronics Co., Ltd. All rights reserved.
#
# Authors: Andrey Moiseenko <moiseenko.a@samsung.com>,
#          Alexander Romanenko <Alexander Romanenko>, 
#          Irina Piontkovskaya <p.irina@samsung.com>, 
#          Manh Ha Le <mh.le@samsung.com>,
#          Mikhail Kudinov <m.kudinov@samsung.com>, 
#          Stepan Muzychka <s.muzychka@samsung.com>  

# src/retrieval-app
# retrieval-app demo application

cmake_minimum_required (VERSION 2.8)

set(CMAKE_VERBOSE_MAKEFILE on)

find_package(CUDA)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch sm_35 )

if( NOT CMAKE_BUILD_TYPE )
 set( CMAKE_BUILD_TYPE Debug)
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    message("Building Debug...")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O0 -g")
elseif(${CMAKE_BUILD_TYPE} STREQUAL "Release")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 --ftz=true --fmad=true")
    message("Building Release...")
endif()

set(TARGET_NAME rnnlm-cuda)
set(rnnlm_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

set(rnnlm_SOURCES
    ${rnnlm_SOURCE_DIR}/common.cpp
    ${rnnlm_SOURCE_DIR}/rnnlmlib.cpp
    ${rnnlm_SOURCE_DIR}/main.cpp
    ${rnnlm_SOURCE_DIR}/cuda-frontend.cpp
    ${rnnlm_SOURCE_DIR}/vocabulary.cpp
    ${rnnlm_SOURCE_DIR}/classic-rnnlm.cpp
    ${rnnlm_SOURCE_DIR}/rnnlm-russian-morphology.cpp
    ${rnnlm_SOURCE_DIR}/cuda-backend.cu
)

set(rnnlm_HEADERS
    ${rnnlm_SOURCE_DIR}/common.h
    ${rnnlm_SOURCE_DIR}/rnnlmlib.h
    ${rnnlm_SOURCE_DIR}/cuda-frontend.h
    ${rnnlm_SOURCE_DIR}/cuda-backend.h
    ${rnnlm_SOURCE_DIR}/vocabulary.h
    ${rnnlm_SOURCE_DIR}/classic-rnnlm.h
    ${rnnlm_SOURCE_DIR}/rnnlm-russian-morphology.h
)

include_directories(
    ${rnnlm_SOURCE_DIR}/common/inc

)

cuda_add_library(${TARGET_NAME} ${LIBRARY_TYPE} ${rnnlm_SOURCES} ${rnnlm_HEADERS})
cuda_add_executable(${TARGET_NAME}-app ${rnnlm_SOURCE_DIR}/main.cpp)
cuda_add_executable(${TARGET_NAME}-test ${rnnlm_SOURCE_DIR}/vocab_test.cpp)
target_link_libraries(${TARGET_NAME} ${CUDA_CUBLAS_LIBRARIES})
target_link_libraries(${TARGET_NAME}-app ${TARGET_NAME})
target_link_libraries(${TARGET_NAME}-test ${TARGET_NAME})
ADD_DEFINITIONS("-Wall" "-fPIC  -std=c++11 -fopenmp")
