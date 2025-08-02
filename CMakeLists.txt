cmake_minimum_required(VERSION 3.15)
project(TensorDemo LANGUAGES CXX)

# ───── Global settings ─────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_policy(SET CMP0135 NEW)          # modern FindPackage behavior

# Optimised release flags
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-O3 -march=native -ffast-math)
elseif (MSVC)
    add_compile_options(/O2 /fp:fast)
endif()

# ───── GoogleTest (fetched at configure time) ──────────────────────
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)   # MSVC static/runtime match
FetchContent_MakeAvailable(googletest)

# Build googletest with default (safe) FP flags
foreach(t gtest gmock gtest_main gmock_main)
    if (TARGET ${t})
        target_compile_options(${t} PRIVATE -fno-fast-math)
    endif()
endforeach()

# ───── Header-only “library” with all tensor sources ───────────────
add_library(tensor_objects INTERFACE)
target_include_directories(tensor_objects INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# ───── Demo executable ─────────────────────────────────────────────
add_executable(tensor_demo src/main.cpp)
target_link_libraries(tensor_demo PRIVATE tensor_objects)

# ───── Unit-test executable ───────────────────────────────────────
enable_testing()

add_executable(unit_tests
    tests/test_tensor_speed.cpp
    tests/test_tensor_intensive.cpp
)

target_link_libraries(unit_tests
    tensor_objects
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(unit_tests DISCOVERY_MODE PRE_TEST)
