cmake_minimum_required(VERSION 3.4.3)

project(rhsass)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
find_package(ComputeCpp REQUIRED)

set(CMAKE_CXX_STANDARD 17)
enable_testing()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
# set(COMPUTECPP_USER_FLAGS "--gcc-toolchain=/usr/local/software/master/gcc/9")

add_executable(rhsass main.cpp)
add_sycl_to_target(TARGET rhsass SOURCES main.cpp)
