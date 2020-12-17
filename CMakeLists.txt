cmake_minimum_required(VERSION 3.6)
project(PaddleHelix)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -fPIC")

add_subdirectory(third-party/pybind11)

set(third_party_deps)
list(APPEND third_party_deps pybind11)

add_subdirectory(c/pahelix/toolkit/linear_rna)


