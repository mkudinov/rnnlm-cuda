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

if (BUILD_MODE)
    if(${BUILD_MODE} STREQUAL DEBUG)
        message("Building DEBUG...")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -ggdb3 -Wno-unknown-pragmas")
    elseif(${BUILD_MODE} STREQUAL RELEASE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -mtune=native -march=native -O3")
        message("Building RELEASE...")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -mtune=native -march=native -ffast-math -O3")
        message("Building DEFAULT...")
    endif()
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -ggdb3 -Wno-unknown-pragmas")
    message("Building DEFAULT...")
endif()


set(TARGET_NAME rnnlm-cuda)
set(rnnlm_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

set(rnnlm_SOURCES
    ${rnnlm_SOURCE_DIR}/rnnlmlib.cpp
    ${rnnlm_SOURCE_DIR}/main.cpp
)

set(rnnlm_HEADERS
    ${rnnlm_SOURCE_DIR}/rnnlmlib.h
)

include_directories(
    ${PROJECT_ROOT_DIR}/CSVManager
    ${PROJECT_ROOT_DIR}/Tools
)


add_library(${TARGET_NAME} ${LIBRARY_TYPE} ${rnnlm_SOURCES} ${rnnlm_HEADERS})
set_target_properties(${TARGET_NAME} PROPERTIES DEFINE_SYMBOL rnnlm_EXPORTS)
add_executable(${TARGET_NAME}-app ${rnnlm_SOURCE_DIR}/main.cpp)
target_link_libraries(${TARGET_NAME} tools csvManager)
target_link_libraries(${TARGET_NAME}-app ${TARGET_NAME})
ADD_DEFINITIONS("-Wall" "-fPIC")
