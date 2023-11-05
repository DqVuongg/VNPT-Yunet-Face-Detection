cmake_minimum_required(VERSION 2.8)



#2.project name
project(Yunet C CXX)
# set(CMAKE_C_COMPILER "/home/vuong/mips-gcc472-glibc216-64bit/bin/mips-linux-uclibc-gnu-gcc")
# set(CMAKE_CXX_COMPILER "/home/vuong/mips-gcc472-glibc216-64bit/bin/mips-linux-uclibc-gnu-g++")

#3.set environment variable
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lpthread -lm")

#4. include
include_directories(${CMAKE_CURRENT_LIST_DIR}/include/ncnn)

#5. ncnn lib
add_library( libncnn STATIC IMPORTED )
set_target_properties(
        libncnn
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/libtest/libncnn.a
)

#6.source directory
file(GLOB TEST_SRC ${CMAKE_CURRENT_LIST_DIR}/main.cpp)
set(TEST_COMPILE_CODE ${TEST_SRC})


add_executable(Yunet ${TEST_COMPILE_CODE})

target_link_libraries(Yunet
                      libncnn
)