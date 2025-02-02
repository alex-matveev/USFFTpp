set (CMAKE_CXX_STANDARD 17)

cmake_minimum_required (VERSION 3.9)

project (USFFTpp VERSION 0.0.1 DESCRIPTION "Unequispaced Fast Fourier Transform implementation")

option(USFFTPP_BUILD_TESTS "Build USFFTpp tests" ON)

add_subdirectory (core)

if(USFFTPP_BUILD_TESTS)
    enable_testing()
    add_subdirectory (tests)
endif()
