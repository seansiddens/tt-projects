include(CTest)
enable_testing()

# add_executable(
#   sample_tests
#   hello_test.cpp
# )
# target_link_libraries(sample_tests
#   GTest::gtest_main
#   sample_lib
# )

# Current tests
add_library(current_lib
    ${CMAKE_SOURCE_DIR}/sources/examples/current/stream.cpp
    ${CMAKE_SOURCE_DIR}/sources/examples/current/kernel.cpp
    ${CMAKE_SOURCE_DIR}/sources/examples/current/map.cpp
)

# Link current_lib with sample_lib
target_link_libraries(current_lib
    PUBLIC
    sample_lib    
)

target_include_directories(current_lib
    PUBLIC
    ${CMAKE_SOURCE_DIR}/sources/examples/current
)

# Create your test executable
add_executable(current_tests
    current_test.cpp
)

# Link against your library
target_link_libraries(current_tests
    PRIVATE
    current_lib
    sample_lib
    GTest::gtest_main
)

include(GoogleTest)
# gtest_discover_tests(sample_tests)
gtest_discover_tests(current_tests)