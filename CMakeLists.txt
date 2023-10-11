cmake_minimum_required(VERSION 3.0)
project(torchcluster)
set(CMAKE_CXX_STANDARD 14)
set(TORCHCLUSTER_VERSION 1.6.3)

option(WITH_CUDA "Enable CUDA support" OFF)
option(WITH_PYTHON "Link to Python when building" ON)

if(WITH_CUDA)
  enable_language(CUDA)
  add_definitions(-D__CUDA_NO_HALF_OPERATORS__)
  add_definitions(-DWITH_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

if (WITH_PYTHON)
  add_definitions(-DWITH_PYTHON)
  find_package(Python3 COMPONENTS Development)
endif()
find_package(Torch REQUIRED)

file(GLOB HEADERS csrc/*.h)
file(GLOB OPERATOR_SOURCES csrc/*.* csrc/cpu/*.*)
if(WITH_CUDA)
  file(GLOB OPERATOR_SOURCES ${OPERATOR_SOURCES} csrc/cuda/*.h csrc/cuda/*.cu)
endif()

add_library(${PROJECT_NAME} SHARED ${OPERATOR_SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
if (WITH_PYTHON)
  target_link_libraries(${PROJECT_NAME} PRIVATE Python3::Python)
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES EXPORT_NAME TorchCluster)

target_include_directories(${PROJECT_NAME} INTERFACE
  "$<BUILD_INTERFACE:${HEADERS}>"
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(TORCHCLUSTER_CMAKECONFIG_INSTALL_DIR "share/cmake/TorchCluster" CACHE STRING "install path for TorchClusterConfig.cmake")

configure_package_config_file(cmake/TorchClusterConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/TorchClusterConfig.cmake"
  INSTALL_DESTINATION ${TORCHCLUSTER_CMAKECONFIG_INSTALL_DIR})

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/TorchClusterConfigVersion.cmake
  VERSION ${TORCHCLUSTER_VERSION}
  COMPATIBILITY AnyNewerVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/TorchClusterConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/TorchClusterConfigVersion.cmake
  DESTINATION ${TORCHCLUSTER_CMAKECONFIG_INSTALL_DIR})

install(TARGETS ${PROJECT_NAME}
  EXPORT TorchClusterTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

install(EXPORT TorchClusterTargets
  NAMESPACE TorchCluster::
  DESTINATION ${TORCHCLUSTER_CMAKECONFIG_INSTALL_DIR})

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})
install(FILES
  csrc/cpu/fps_cpu.h
  csrc/cpu/graclus_cpu.h
  csrc/cpu/grid_cpu.h
  csrc/cpu/rw_cpu.h
  csrc/cpu/sampler_cpu.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/cpu)
if(WITH_CUDA)
  install(FILES
    csrc/cuda/fps_cuda.h
    csrc/cuda/graclus_cuda.h
    csrc/cuda/grid_cuda.h
    csrc/cuda/knn_cuda.h
    csrc/cuda/nearest_cuda.h
    csrc/cuda/radius_cuda.h
    csrc/cuda/rw_cuda.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/cuda)
endif()

if(WITH_CUDA)
  set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")
  set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")
endif()
