project (gather)

set(SOURCES main.cpp)

add_executable(gather ${SOURCES})
target_link_libraries(gather PRIVATE sample_lib)