project (branch_test)

set(SOURCES main.cpp)

add_executable(branch_test ${SOURCES})
target_link_libraries(branch_test PRIVATE sample_lib)