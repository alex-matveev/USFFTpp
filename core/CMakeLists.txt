include(GNUInstallDirs)

add_library (USFFTpp SHARED src/usfftpp.cpp)

set(USFFTPP_PUBLIC_HEADERS include/usfftpp.h include/usfft-seq.h include/convPolicies.h)

set_target_properties(USFFTpp PROPERTIES 
	VERSION ${PROJECT_VERSION}
	SOVERSION ${PROJECT_VERSION_MAJOR}
	PUBLIC_HEADER "${USFFTPP_PUBLIC_HEADERS}")

target_include_directories (USFFTpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(USFFTpp PRIVATE USFFTPP_EXPORT)

find_package (OpenMP)

if (OpenMP_CXX_FOUND)
	message(STATUS "${OpenMP_CXX_LIBRARIES}")
endif (OpenMP_CXX_FOUND)

target_link_libraries (USFFTpp LINK_PUBLIC m dl OpenMP::OpenMP_CXX fftw3f fftw3)

install(TARGETS USFFTpp
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/usfftpp)
