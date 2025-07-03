cmake_minimum_required(VERSION 3.15)
project(gaticc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_EXTENSIONS OFF)

set(EXE_NAME gaticc)
set(LIB_NAME gati)
set(LIB_NAME_FULL libgati.so)
set(PYLIB_NAME _gati)
set(PCH_LIBS ${CMAKE_CURRENT_SOURCE_DIR}/src/pch.h)
set(BOOST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/boost/)
set(PROTO_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/protobuf/)
set(PYBIND_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11/)
set(GATI_LIB_PATH ${CMAKE_CURRENT_BINARY_DIR}/${LIB_NAME_FULL})
set(PROTO_BIN_PATH ${CMAKE_CURRENT_BINARY_DIR}/protobuf)
set(PROTOC_EXE ${CMAKE_CURRENT_BINARY_DIR}/protobuf/protoc)
set(PARENT_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR})
set(EXTENDED_OPTS -Wall -Wextra -pedantic -Wno-narrowing)
add_compile_definitions(PRINT_COLOR)

# Build time configurable options
# Instead of enabling/disabling options in this file, consider
# providing them from the command line 
if (NOT BUILD_TESTING)
  set(BUILD_TESTING OFF)
endif()
message(STATUS "Tests have been set to ${BUILD_TESTING}")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type has been set to ${CMAKE_BUILD_TYPE}")

if (NOT CMAKE_VERBOSE_MAKEFILE)
  set(CMAKE_VERBOSE_MAKEFILE OFF)
endif()
message(STATUS "Makefile verbosity set to ${CMAKE_VERBOSE_MAKEFILE}")

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(PROTO_LIB_NAME libprotobuf-lited)
else()
  set(PROTO_LIB_NAME libprotobuf-lite)
endif()
message(STATUS "Proto library name set to ${PROTO_LIB_NAME}")


set(ALL_DIR
  ${Python3_INCLUDE_DIRS}
  ${BOOST_PATH}
  ${PROTO_PATH}/src
  ${PYBIND_PATH}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}
)

set(ALL_LIBS
  ${Python3_LIBRARIES} 
  ${PROTO_BIN_PATH}/${PROTO_LIB_NAME}.so
  dl
)

add_subdirectory(${PYBIND_PATH})

set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build tests")
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libs")
add_subdirectory(${PROTO_PATH} ${PROTO_BIN_PATH})
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.cc ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.h
  COMMAND ${PROTOC_EXE} --cpp_out=${CMAKE_CURRENT_BINARY_DIR} --proto_path=${CMAKE_CURRENT_SOURCE_DIR} onnx.proto
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/onnx.proto protoc
)
add_custom_target(pbgen DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.cc ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.h)

file(GLOB SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
list(REMOVE_ITEM FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

add_library(${LIB_NAME} SHARED ${FILES} ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.cc)
target_include_directories(${LIB_NAME} PUBLIC ${ALL_DIR})
target_link_libraries(${LIB_NAME} ${ALL_LIBS})
target_compile_options(${LIB_NAME} PRIVATE ${EXTENDED_OPTS})
add_dependencies(${LIB_NAME} libprotobuf-lite pbgen)

add_executable(${EXE_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp) 
set_target_properties(${EXE_NAME} PROPERTIES INSTALL_RPATH "/usr/local/lib")
target_precompile_headers(${EXE_NAME} PRIVATE ${PCH_LIBS})
target_link_libraries(${EXE_NAME} ${LIB_NAME} ${ALL_LIBS})
target_include_directories(${EXE_NAME} PUBLIC ${ALL_DIR})
target_compile_options(${EXE_NAME} PRIVATE ${EXTENDED_OPTS})
add_dependencies(${EXE_NAME} ${LIB_NAME} libprotobuf-lite pbgen)

pybind11_add_module(${PYLIB_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp)
set_target_properties(${PYLIB_NAME} PROPERTIES INSTALL_RPATH "/usr/local/lib")
target_link_libraries(${PYLIB_NAME} PRIVATE ${GATI_LIB_PATH} ${ALL_LIBS})
target_include_directories(${PYLIB_NAME} PRIVATE ${ALL_DIR})
add_dependencies(${PYLIB_NAME} ${LIB_NAME} libprotobuf-lite pbgen)

include(GNUInstallDirs)
install(TARGETS ${LIB_NAME}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(TARGETS ${EXE_NAME}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

execute_process(
  COMMAND python3 -c "import site, sys; print(site.getsitepackages()[0] if hasattr(site, 'getsitepackages') else site.USER_SITE)"
  OUTPUT_VARIABLE PYTHON_SITE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

install(TARGETS ${PYLIB_NAME}
  LIBRARY DESTINATION ${PYTHON_SITE_DIR}
)

if (BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
