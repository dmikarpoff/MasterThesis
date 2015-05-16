cmake_minimum_required(VERSION 2.8)

project(Matching)

set(CMAKE_BUILD_TYPE Release)

add_definitions(-std=c++11)

include_directories(${OpenCV_INCLUDE_DIRS})

find_package(OpenCV 2.4.10 EXACT REQUIRED core highgui features2d nonfree)


add_executable(matcher               main.cpp)

target_link_libraries(matcher        ${OpenCV_LIBS})
