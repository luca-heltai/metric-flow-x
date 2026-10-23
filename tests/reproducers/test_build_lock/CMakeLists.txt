cmake_minimum_required(VERSION 3.23)
project(TestBuildLock NONE)
enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
option(APPLY_LOCK "Include the project's CTest build lock" ON)

add_custom_target(shared_library_dummy
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/run.py"
    --worker "${CMAKE_CURRENT_BINARY_DIR}"
  VERBATIM)

macro(deal_II_pickup_tests)
  foreach(i RANGE 1 4)
    add_custom_target(dummy_${i} DEPENDS shared_library_dummy)
    add_test(NAME nested_ninja_${i}
      COMMAND "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}" --target dummy_${i})
  endforeach()
endmacro()

if(APPLY_LOCK)
  include("${PROJECT_TEST_CMAKE}")
else()
  deal_II_pickup_tests()
endif()
