cmake_minimum_required(VERSION 3.6)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/linear_x")

file(GLOB src
    "${CMAKE_CURRENT_SOURCE_DIR}/linear_x/*/*.cpp")
add_library(linear_x_core STATIC ${src})

pybind11_add_module(linear_x ${CMAKE_CURRENT_SOURCE_DIR}/linear_x/linear_x.cpp)
add_dependencies(linear_x ${third_party_deps} linear_x_core)
target_link_libraries(linear_x PRIVATE pybind11::module linear_x_core)
