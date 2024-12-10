cmake_minimum_required(VERSION 3.1.0)
project(robust-laplacian-py-ext)

add_subdirectory(deps/pybind11)

# set location of eigen for geometry-central
set(GC_EIGEN_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/deps/eigen" CACHE PATH "my path")

# geometry-central
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(deps/geometry-central)


pybind11_add_module(robust_laplacian_bindings_ext 
  cpp/point_cloud_utilities.cpp
  cpp/core.cpp
  cpp/mesh_utilities.cpp
  cpp/geometrycentralext/heat_method_distance_ext.cpp
  cpp/geometrycentralext/point_cloud_heat_solver_ext.cpp
  cpp/geometrycentralext/neighborhoods_ext.cpp
  cpp/geometrycentralext/point_position_geometry_ext.cpp
  cpp/geometrycentralext/point_cloud_io_ext.cpp
  cpp/geometrycentralext/local_triangulation_ext.cpp
)

include_directories(robust_laplacian_bindings_ext ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
include_directories(robust_laplacian_bindings_ext ${CMAKE_CURRENT_SOURCE_DIR}/deps/jc_voronoi/include)

target_link_libraries(robust_laplacian_bindings_ext PRIVATE geometry-central)