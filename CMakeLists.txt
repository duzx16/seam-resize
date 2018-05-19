cmake_minimum_required(VERSION 3.7)
project(seam_resize)
find_package(OpenCV)
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES main.cpp)

add_executable(seam_resize ${SOURCE_FILES})
target_link_libraries(seam_resize ${OpenCV_LIBS} )