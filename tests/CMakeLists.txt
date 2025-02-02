FIND_PACKAGE(GTest REQUIRED)

add_executable (USFFTpp-test tests.cpp)
add_executable (USFFTpp-test2d tests2d.cpp)

target_link_libraries (USFFTpp-test LINK_PUBLIC USFFTpp GTest::gtest_main)
target_link_libraries (USFFTpp-test2d LINK_PUBLIC USFFTpp GTest::gtest_main)

add_test(NAME USFFTpp-test1d COMMAND USFFTpp-test)
add_test(NAME USFFTpp-test2d COMMAND USFFTpp-test2d)
