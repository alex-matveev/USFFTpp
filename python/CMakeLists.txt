cmake_minimum_required(VERSION 3.9)

project(USFFTpp-python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

python_add_library(_usfftpp MODULE WITH_SOABI bindings.cpp)
target_link_libraries(_usfftpp PRIVATE USFFTpp::USFFTpp pybind11::module)

target_include_directories(_usfftpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../core/include
)

set_target_properties(_usfftpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/usfftpp
)

if (SKBUILD)
    set(USFFTPP_PYTHON_INSTALL_DIR usfftpp)
else()
    set(USFFTPP_PYTHON_INSTALL_DIR
        lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/usfftpp)
endif()

install(TARGETS _usfftpp
        LIBRARY DESTINATION ${USFFTPP_PYTHON_INSTALL_DIR})

install(
    DIRECTORY usfftpp/
    DESTINATION ${USFFTPP_PYTHON_INSTALL_DIR}
    FILES_MATCHING PATTERN "*.py"
)

if(NOT USFFTPP_BUILD_TESTS)
    return()
endif()

add_test(
    NAME python.usfftpp.basic
    COMMAND ${CMAKE_COMMAND} -E env OMP_NUM_THREADS=1
            ${Python_EXECUTABLE} -m unittest discover -s tests -p "test_*.py"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

