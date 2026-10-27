cmake_minimum_required(VERSION 3.23)

project(MetricFlowX VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(deal.II 9.5.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
  )
ENDIF()

# Pin the public vector/matrix ABI for both this build and installed consumers.
set(METRIC_FLOW_X_BACKEND "AUTO" CACHE STRING "Linear algebra backend: AUTO, PETSc, or Trilinos")
set_property(CACHE METRIC_FLOW_X_BACKEND PROPERTY STRINGS AUTO PETSc Trilinos)
string(TOUPPER "${METRIC_FLOW_X_BACKEND}" _backend)
if(_backend STREQUAL "AUTO")
  if(DEAL_II_WITH_PETSC AND NOT DEAL_II_PETSC_WITH_COMPLEX)
    set(_backend "PETSC")
  else()
    set(_backend "TRILINOS")
  endif()
endif()
if(_backend STREQUAL "PETSC")
  if(NOT DEAL_II_WITH_PETSC OR DEAL_II_PETSC_WITH_COMPLEX)
    message(FATAL_ERROR "MetricFlowX PETSc backend requires deal.II with real-valued PETSc")
  endif()
  set(METRIC_FLOW_X_WITH_PETSC ON)
elseif(_backend STREQUAL "TRILINOS")
  if(NOT DEAL_II_WITH_TRILINOS)
    message(FATAL_ERROR "MetricFlowX Trilinos backend requires deal.II with Trilinos")
  endif()
  set(METRIC_FLOW_X_WITH_TRILINOS ON)
else()
  message(FATAL_ERROR "METRIC_FLOW_X_BACKEND must be AUTO, PETSc, or Trilinos")
endif()
message(STATUS "MetricFlowX linear algebra backend: ${_backend}")
configure_file(include/metric_flow_x/config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/metric_flow_x/config.h)

# Preserve DebugRelease before deal.II can normalize CMAKE_BUILD_TYPE.
set(_requested_build_type "${CMAKE_BUILD_TYPE}")
if(_requested_build_type AND NOT _requested_build_type MATCHES "^(Debug|Release|DebugRelease)$")
  message(FATAL_ERROR "Supported CMAKE_BUILD_TYPE values: Debug, Release, DebugRelease")
endif()
deal_II_initialize_cached_variables()
if(_requested_build_type)
  set(CMAKE_BUILD_TYPE "${_requested_build_type}" CACHE STRING "Build type" FORCE)
endif()
if(CMAKE_CONFIGURATION_TYPES)
  message(FATAL_ERROR "Use a single-config generator for MetricFlowX (e.g. Ninja or Unix Makefiles)")
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release DebugRelease)
if(CMAKE_BUILD_TYPE STREQUAL "DebugRelease")
  set(METRIC_FLOW_X_BUILD_TYPES Release Debug)
else()
  set(METRIC_FLOW_X_BUILD_TYPES "${CMAKE_BUILD_TYPE}")
endif()
foreach(_variant IN LISTS METRIC_FLOW_X_BUILD_TYPES)
  string(TOUPPER "${_variant}" _variant_upper)
  if(NOT _variant_upper IN_LIST DEAL_II_BUILD_TYPES)
    message(FATAL_ERROR "deal.II does not provide ${_variant}; available variants: ${DEAL_II_BUILD_TYPES}")
  endif()
endforeach()

set(METRIC_FLOW_X_SOURCES
  source/blood_flow_system.cc
  source/vessel_partitioning.cc
  source/viscous_wall.cc
  source/blood_flow_initialization.cc
  source/inflow_series.cc
  source/steady_network.cc
  source/io/vtk_utils.cc
  source/parsed_tools/constants.cc
  source/parsed_tools/function.cc
)

set(METRIC_FLOW_X_INSTALL_TARGETS)
foreach(_variant IN LISTS METRIC_FLOW_X_BUILD_TYPES)
  string(TOUPPER "${_variant}" _variant_upper)
  set(_suffix "")
  if(_variant STREQUAL "Debug" AND CMAKE_BUILD_TYPE STREQUAL "DebugRelease")
    set(_suffix "_debug")
  endif()
  set(_library "metric_flow_x${_suffix}")
  set(_application "blood_flow${_suffix}")
  add_library(${_library} SHARED ${METRIC_FLOW_X_SOURCES})
  target_include_directories(${_library} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_include_directories(${_library} SYSTEM PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/rapidcsv")
  deal_ii_setup_target(${_library} ${_variant_upper})
  set_target_properties(${_library} PROPERTIES EXPORT_NAME metric_flow_x_${_variant})

  add_executable(${_application} apps/blood_flow.cc)
  target_compile_definitions(${_application} PRIVATE
    PRM_DIR="${CMAKE_CURRENT_BINARY_DIR}/parameters/")
  deal_ii_setup_target(${_application} ${_variant_upper})
  target_link_libraries(${_application} ${_library})
  file(RELATIVE_PATH _library_relative_to_bin
    "${CMAKE_INSTALL_FULL_BINDIR}" "${CMAKE_INSTALL_FULL_LIBDIR}")
  if(APPLE)
    set_property(TARGET ${_application} APPEND PROPERTY INSTALL_RPATH
      "@loader_path/${_library_relative_to_bin}")
  elseif(UNIX)
    set_property(TARGET ${_application} APPEND PROPERTY INSTALL_RPATH
      "$ORIGIN/${_library_relative_to_bin}")
  endif()
  # Retain runtime search paths for shared deal.II dependencies as well.
  set_target_properties(${_library} ${_application} PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE)
  set(TEST_LIBRARIES_${_variant_upper} ${_library})
  list(APPEND METRIC_FLOW_X_INSTALL_TARGETS ${_library} ${_application})
endforeach()

# The build-tree alias is the default variant; tests use explicit variants.
add_library(MetricFlowX::metric_flow_x ALIAS metric_flow_x)


# Mirror parameters/ into the build tree so applications always use
# configured files. Both .prm and .prm.in expand the source tree location;
# the .in suffix is removed from template destinations.
set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
file(GLOB_RECURSE PARAMETER_FILES CONFIGURE_DEPENDS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/parameters"
  "${CMAKE_CURRENT_SOURCE_DIR}/parameters/*")
foreach(parameter_file IN LISTS PARAMETER_FILES)
  set(parameter_source "${CMAKE_CURRENT_SOURCE_DIR}/parameters/${parameter_file}")
  set(parameter_destination "${CMAKE_CURRENT_BINARY_DIR}/parameters/${parameter_file}")
  get_filename_component(parameter_destination_dir "${parameter_destination}" DIRECTORY)
  file(MAKE_DIRECTORY "${parameter_destination_dir}")

  if(parameter_file MATCHES "\\.prm\\.in$")
    string(REGEX REPLACE "\\.in$" "" parameter_destination "${parameter_destination}")
    configure_file("${parameter_source}" "${parameter_destination}" @ONLY)
  elseif(parameter_file MATCHES "\\.prm$")
    configure_file("${parameter_source}" "${parameter_destination}" @ONLY)
  else()
    configure_file("${parameter_source}" "${parameter_destination}" COPYONLY)
  endif()
endforeach()

set(METRIC_FLOW_X_PARAMETER_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/parameters/")
add_compile_definitions(
  METRIC_FLOW_X_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

add_executable(benchmark_assembly EXCLUDE_FROM_ALL tools/benchmark_assembly.cc)
list(GET METRIC_FLOW_X_BUILD_TYPES 0 _benchmark_variant)
string(TOUPPER "${_benchmark_variant}" _benchmark_variant_upper)
deal_ii_setup_target(benchmark_assembly ${_benchmark_variant_upper})
target_link_libraries(benchmark_assembly metric_flow_x)

add_executable(adavn_mean_flow EXCLUDE_FROM_ALL tools/adan/mean_flow.cc)
target_include_directories(adavn_mean_flow SYSTEM PRIVATE third_party/rapidcsv)
deal_ii_setup_target(adavn_mean_flow ${_benchmark_variant_upper})
target_link_libraries(adavn_mean_flow metric_flow_x)

# Add tests
enable_testing()
add_subdirectory(tests)

install(TARGETS ${METRIC_FLOW_X_INSTALL_TARGETS}
  EXPORT MetricFlowXTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/metric_flow_x
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PATTERN "*.in" EXCLUDE
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/metric_flow_x/config.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/metric_flow_x
)

configure_package_config_file(
  cmake/metric_flow_x_config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/MetricFlowXConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MetricFlowX
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/MetricFlowXConfigVersion.cmake
  COMPATIBILITY SameMajorVersion
)

install(EXPORT MetricFlowXTargets
  FILE MetricFlowXTargets.cmake
  NAMESPACE MetricFlowX::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MetricFlowX
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/MetricFlowXConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/MetricFlowXConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MetricFlowX
)
