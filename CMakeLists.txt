cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(automated_alignment)

find_package(PCL 1.6 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (automated_alignment automated_alignment.cpp)
target_link_libraries (automated_alignment ${PCL_LIBRARIES})
