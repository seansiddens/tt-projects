project (leftover_write)

set(SOURCES main.cpp)

add_executable(leftover_write ${SOURCES})
target_link_libraries(leftover_write PRIVATE sample_lib)