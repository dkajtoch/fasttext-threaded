cmake_minimum_required(VERSION 3.18)

project(fasttext_threaded LANGUAGES CXX)

option(FASTTEXT_THREADED_BUILD_TESTS "Build C++ tests" OFF)
option(FASTTEXT_THREADED_ENABLE_ASAN "Build with AddressSanitizer" OFF)
option(FASTTEXT_THREADED_ENABLE_TSAN "Build with ThreadSanitizer" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_CMAKEDIR_RESULT
  )
  if(NOT PYBIND11_CMAKEDIR_RESULT EQUAL 0)
    message(FATAL_ERROR "pybind11 is required but its CMake package was not found")
  endif()
  set(pybind11_DIR ${PYBIND11_CMAKE_DIR})
  find_package(pybind11 CONFIG REQUIRED)
endif()

set(FASTTEXT_VENDOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpp/vendor/fasttext)
set(FASTTEXT_VENDOR_SOURCES
  ${FASTTEXT_VENDOR_DIR}/src/args.cc
  ${FASTTEXT_VENDOR_DIR}/src/autotune.cc
  ${FASTTEXT_VENDOR_DIR}/src/densematrix.cc
  ${FASTTEXT_VENDOR_DIR}/src/dictionary.cc
  ${FASTTEXT_VENDOR_DIR}/src/fasttext.cc
  ${FASTTEXT_VENDOR_DIR}/src/loss.cc
  ${FASTTEXT_VENDOR_DIR}/src/matrix.cc
  ${FASTTEXT_VENDOR_DIR}/src/meter.cc
  ${FASTTEXT_VENDOR_DIR}/src/model.cc
  ${FASTTEXT_VENDOR_DIR}/src/productquantizer.cc
  ${FASTTEXT_VENDOR_DIR}/src/quantmatrix.cc
  ${FASTTEXT_VENDOR_DIR}/src/utils.cc
  ${FASTTEXT_VENDOR_DIR}/src/vector.cc
)

add_library(fasttext_vendor STATIC ${FASTTEXT_VENDOR_SOURCES})
target_compile_features(fasttext_vendor PUBLIC cxx_std_17)
target_include_directories(fasttext_vendor PUBLIC ${FASTTEXT_VENDOR_DIR}/src)
set_target_properties(fasttext_vendor PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

if(FASTTEXT_THREADED_ENABLE_ASAN AND FASTTEXT_THREADED_ENABLE_TSAN)
  message(FATAL_ERROR "AddressSanitizer and ThreadSanitizer cannot be enabled together")
endif()

set(FASTTEXT_THREADED_SANITIZER_FLAGS "")
if(FASTTEXT_THREADED_ENABLE_ASAN)
  set(FASTTEXT_THREADED_SANITIZER_FLAGS "-fsanitize=address,undefined")
elseif(FASTTEXT_THREADED_ENABLE_TSAN)
  set(FASTTEXT_THREADED_SANITIZER_FLAGS "-fsanitize=thread")
endif()

if(FASTTEXT_THREADED_SANITIZER_FLAGS)
  foreach(target fasttext_vendor)
    target_compile_options(${target} PRIVATE
      ${FASTTEXT_THREADED_SANITIZER_FLAGS}
      -fno-omit-frame-pointer
    )
    target_link_options(${target} PRIVATE ${FASTTEXT_THREADED_SANITIZER_FLAGS})
  endforeach()
endif()

set(FASTTEXT_THREADED_SOURCES
  cpp/src/bindings.cpp
  cpp/src/parallel_model.cpp
  cpp/src/thread_pool.cpp
)

pybind11_add_module(_fasttext_threaded MODULE ${FASTTEXT_THREADED_SOURCES})
target_compile_features(_fasttext_threaded PRIVATE cxx_std_17)
target_include_directories(_fasttext_threaded PRIVATE
  cpp/include
)
target_link_libraries(_fasttext_threaded PRIVATE fasttext_vendor)
set_target_properties(_fasttext_threaded PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

if(FASTTEXT_THREADED_SANITIZER_FLAGS)
  target_compile_options(_fasttext_threaded PRIVATE
    ${FASTTEXT_THREADED_SANITIZER_FLAGS}
    -fno-omit-frame-pointer
  )
  target_link_options(_fasttext_threaded PRIVATE ${FASTTEXT_THREADED_SANITIZER_FLAGS})
endif()

if(UNIX AND NOT APPLE)
  target_link_libraries(_fasttext_threaded PRIVATE pthread)
endif()

install(TARGETS _fasttext_threaded LIBRARY DESTINATION fasttext_threaded)

if(FASTTEXT_THREADED_BUILD_TESTS)
  enable_testing()
  add_subdirectory(cpp/tests)
endif()
