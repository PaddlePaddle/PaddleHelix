cmake_minimum_required(VERSION 3.6)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/linear_rna")

file(GLOB src
    "${CMAKE_CURRENT_SOURCE_DIR}/linear_rna/*/*.cpp")
add_library(linear_rna_core STATIC ${src})

pybind11_add_module(linear_rna ${CMAKE_CURRENT_SOURCE_DIR}/linear_rna/linear_rna.cpp)
add_dependencies(linear_rna ${third_party_deps} linear_rna_core)
target_link_libraries(linear_rna PRIVATE pybind11::module linear_rna_core)
