project (leftover_test)

set(SOURCES main.cpp)

add_executable(leftover_test ${SOURCES})
target_link_libraries(leftover_test PRIVATE sample_lib)