cmake_minimum_required(VERSION 3.20)

project(QWEN3_600M LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# GPU architectures:
# 75=Turing (20xx), 
# 86=Ampere (30xx), 
# 89=Ada Lovelace (40xx), 
# 90=Hopper
set(CMAKE_CUDA_ARCHITECTURES 75 86 89 90)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message(STATUS "========================")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Targeting CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

add_executable(
    qwen600
    main.cu
)

target_link_libraries(
    qwen600 
    PRIVATE
    CUDA::cublas
    # CUDA::cudart
    )

target_compile_options(qwen600 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

set_target_properties(qwen600 PROPERTIES COMPILE_FLAGS "-O3")

# disable all warnings for both C++ and CUDA
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(qwen600 PRIVATE -w)
endif()

if (CMAKE_CUDA_COMPILER_ID MATCHES "NVIDIA")
    target_compile_options(qwen600 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-w>)
endif()
