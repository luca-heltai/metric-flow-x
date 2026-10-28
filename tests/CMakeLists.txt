# Library regressions use fixed inputs without standalone IDA-runner controls.
# Keep these separate from the evolving tutorial and paper startup policy.
foreach(_fixture adan56 constant aortic multi_vessel wall_arterial wall_venous wall_mixed)
  configure_file("parameters/${_fixture}.prm.in"
    "parameters/${_fixture}.prm" @ONLY)
endforeach()
add_compile_definitions(PRM_DIR="${CMAKE_CURRENT_BINARY_DIR}/parameters/")

include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/third_party/rapidcsv")
deal_II_pickup_tests()

# Each deal.II test starts a build in the same binary directory. Ninja's
# dependency database and shared library outputs require a single writer.
get_property(_metric_flow_x_tests DIRECTORY PROPERTY TESTS)
if(_metric_flow_x_tests)
  set_tests_properties(${_metric_flow_x_tests} PROPERTIES
    RESOURCE_LOCK metric_flow_x_test_build)
endif()
