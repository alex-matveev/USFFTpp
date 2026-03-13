set (CMAKE_CXX_STANDARD 20)

cmake_minimum_required (VERSION 3.9)

project (USFFTpp VERSION 0.0.1 DESCRIPTION "Unequispaced Fast Fourier Transform implementation")

option(USFFTPP_BUILD_TESTS "Build USFFTpp tests" ON)
option(USFFTPP_BUILD_PYTHON "Build USFFTpp Python bindings" ON)

add_subdirectory(core)

if(USFFTPP_BUILD_TESTS)
    enable_testing()
    add_subdirectory (tests)
endif()

if(USFFTPP_BUILD_PYTHON)
    add_subdirectory(python)
endif()
