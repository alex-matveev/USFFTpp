include(GNUInstallDirs)

add_library (USFFTpp src/usfftpp.cpp)
add_library(USFFTpp::USFFTpp ALIAS USFFTpp)

set_target_properties(USFFTpp PROPERTIES 
	VERSION ${PROJECT_VERSION}
	SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON)

target_include_directories(USFFTpp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_compile_definitions(USFFTpp PRIVATE USFFTPP_EXPORT)

find_package(OpenMP)

if (OpenMP_CXX_FOUND)
	message(STATUS "${OpenMP_CXX_LIBRARIES}")
endif (OpenMP_CXX_FOUND)

target_link_libraries(USFFTpp PRIVATE OpenMP::OpenMP_CXX fftw3f fftw3)

install(TARGETS USFFTpp
	EXPORT USFFTppTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT USFFTppTargets
    FILE USFFTppTargets.cmake
    NAMESPACE USFFTpp::
    DESTINATION lib/cmake/USFFTpp
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    ../cmake/USFFTppConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/USFFTppConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/USFFTpp
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/USFFTppConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/USFFTppConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/USFFTppConfigVersion.cmake
    DESTINATION lib/cmake/USFFTpp
)
