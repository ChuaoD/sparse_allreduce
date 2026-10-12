cmake_minimum_required(VERSION 3.14)
project(sparse_allreduce LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_package(MPI REQUIRED COMPONENTS CXX)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

# Sparse collective shared by the benchmark and profiling drivers
add_library(sparse_allreduce STATIC src/sparse_allreduce.cpp src/sparse_allreduce.h)
target_include_directories(sparse_allreduce PUBLIC src)
target_link_libraries(sparse_allreduce PUBLIC MPI::MPI_CXX)

add_executable(mpi main.cpp)
target_link_libraries(mpi PRIVATE sparse_allreduce)

add_executable(mpi_runtime runtime/main_runtime.cpp)
target_link_libraries(mpi_runtime PRIVATE sparse_allreduce)
