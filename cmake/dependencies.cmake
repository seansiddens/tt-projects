
set(ENV{CPM_SOURCE_CACHE} "${PROJECT_SOURCE_DIR}/.cpmcache")

############################################################################################################################
# Boost
############################################################################################################################

include(${PROJECT_SOURCE_DIR}/cmake/fetch_boost.cmake)

fetch_boost_library(smart_ptr)

############################################################################################################################
# yaml-cpp
############################################################################################################################

CPMAddPackage(
  NAME yaml-cpp
  GITHUB_REPOSITORY jbeder/yaml-cpp
  GIT_TAG 0.8.0
  OPTIONS
    "YAML_CPP_BUILD_TESTS OFF"
    "YAML_CPP_BUILD_TOOLS OFF"
    "YAML_BUILD_SHARED_LIBS OFF"
)

if (yaml-cpp_ADDED)
  target_link_libraries(yaml-cpp PUBLIC ${LIBC++} ${LIBC++ABI})
  target_compile_options(yaml-cpp PUBLIC -stdlib=libc++)
endif()

############################################################################################################################
# googletest
############################################################################################################################

CPMAddPackage(
  NAME googletest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG v1.13.0
  VERSION 1.13.0
  OPTIONS "INSTALL_GTEST OFF"
)

if (googletest_ADDED)
    target_compile_options(gtest PRIVATE -Wno-implicit-int-float-conversion)
    target_link_libraries(gtest PUBLIC ${LIBC++} ${LIBC++ABI})
    target_compile_options(gtest PUBLIC -stdlib=libc++)
    target_link_libraries(gtest_main PUBLIC ${LIBC++} ${LIBC++ABI})
    target_compile_options(gtest_main PUBLIC -stdlib=libc++)
endif()

############################################################################################################################
# boost-ext reflect : https://github.com/boost-ext/reflect
############################################################################################################################

CPMAddPackage(
  NAME reflect
  GITHUB_REPOSITORY boost-ext/reflect
  GIT_TAG v1.1.1
)


CPMAddPackage(
  NAME googletest
  GITHUB_REPOSITORY google/googletest
  GIT_TAG v1.13.0
  VERSION 1.13.0
  OPTIONS "INSTALL_GTEST OFF"
)

CPMAddPackage(
    NAME fmt
    GITHUB_REPOSITORY fmtlib/fmt
    GIT_TAG 10.1.1  # You can adjust version as needed
    OPTIONS 
        "FMT_INSTALL OFF"
        "FMT_TEST OFF"
        "FMT_DOC OFF"
        "FMT_SYSTEM_HEADERS ON"
)
