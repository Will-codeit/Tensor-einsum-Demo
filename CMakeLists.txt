

cmake_minimum_required(VERSION 3.16)   
project(TensorDemo LANGUAGES CXX)
cmake_policy(SET CMP0135 NEW)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ────────────────────── Google Test ──────────────────────
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ─────────────────── Header-only “library” ───────────────
add_library(tensor_objects INTERFACE)
target_include_directories(tensor_objects INTERFACE src)

# ───────────────────── Demo executable ───────────────────
add_executable(demo src/main.cpp)

target_link_libraries(demo PRIVATE tensor_objects)

# ───────────────────── Unit-test target ──────────────────
enable_testing()
add_executable(unit_tests
    tests/test_tensor.cpp
    tests/test_tensor_intensive.cpp      
    tests/test_tensor_speed.cpp
)
target_link_libraries(unit_tests
    tensor_objects
    GTest::gtest_main
)
include(GoogleTest)
gtest_discover_tests(unit_tests)