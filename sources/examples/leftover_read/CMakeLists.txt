project (leftover_read)

set(SOURCES main.cpp)

add_executable(leftover_read ${SOURCES})
target_link_libraries(leftover_read PRIVATE sample_lib)