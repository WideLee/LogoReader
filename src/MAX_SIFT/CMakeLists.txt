cmake_minimum_required(VERSION 3.7)
project(MAX_SIFT)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV)

set(SOURCE_FILES
        max_sift.hpp
        test_max_sift.hpp
        testMain.cpp)

add_executable(MAX_SIFT ${SOURCE_FILES})

target_link_libraries(MAX_SIFT ${OpenCV_LIBS})
