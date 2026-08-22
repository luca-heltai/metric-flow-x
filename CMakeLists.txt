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

deal_II_initialize_cached_variables()

set(METRIC_FLOW_X_SOURCES
  source/blood_flow_system.cc
  source/io/vtk_utils.cc
  source/parsed_tools/constants.cc
  source/parsed_tools/function.cc
)

add_library(metric_flow_x SHARED ${METRIC_FLOW_X_SOURCES})
add_library(MetricFlowX::metric_flow_x ALIAS metric_flow_x)

target_include_directories(metric_flow_x
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
deal_ii_setup_target(metric_flow_x)


# Mirror parameters/ into the build tree so tests and examples always use
# configured files.  Template files end in .prm.in and are expanded with the
# source tree location baked in.
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
  else()
    configure_file("${parameter_source}" "${parameter_destination}" COPYONLY)
  endif()
endforeach()

add_executable(blood_flow apps/blood_flow.cc)
target_compile_definitions(blood_flow PRIVATE
  PRM_DIR="${CMAKE_CURRENT_BINARY_DIR}/parameters/")
deal_ii_setup_target(blood_flow)
target_link_libraries(blood_flow MetricFlowX::metric_flow_x)

set(TEST_LIBRARIES MetricFlowX::metric_flow_x)
set(METRIC_FLOW_X_PARAMETER_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/parameters/")
add_compile_definitions(PRM_DIR="${CMAKE_CURRENT_BINARY_DIR}/parameters/")

# Add tests
add_subdirectory(tests)
enable_testing()

install(TARGETS metric_flow_x blood_flow
  EXPORT MetricFlowXTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/metric_flow_x
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

configure_package_config_file(
  cmake/MetricFlowXConfig.cmake.in
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
