cmake_minimum_required(VERSION 3.15)

project(Testing)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()
set(TESTS 
  test_file
)

foreach(TEST ${TESTS})
    add_executable(${TEST} ${TEST}.cpp)
    add_dependencies(${TEST} libprotobuf-lite gati pbgen)
    target_include_directories(${TEST} PRIVATE ${ALL_DIR})
    target_link_libraries(${TEST} PRIVATE 
      ${LIB_NAME}
      ${ALL_LIBS}
    )
    add_test(NAME ${TEST} COMMAND ${TEST})
endforeach()

add_custom_target(tests DEPENDS ${TESTS} ${PROTO_LIB_NAME} gati)
