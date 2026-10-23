include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/third_party/rapidcsv")
deal_II_pickup_tests()

# Each deal.II test starts a build in the same binary directory. Ninja's
# dependency database and shared library outputs require a single writer.
get_property(_metric_flow_x_tests DIRECTORY PROPERTY TESTS)
if(_metric_flow_x_tests)
  set_tests_properties(${_metric_flow_x_tests} PROPERTIES
    RESOURCE_LOCK metric_flow_x_test_build)
endif()
