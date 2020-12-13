cmake_minimum_required(VERSION 3.5.0)

# set(CMAKE_CXX_STANDARD 17)
project(OCR)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++14 ")

find_package(OpenCV 3.0 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# Find all executables
file(GLOB project_SRCS src/*.cpp) 

# Add project executable
add_executable(detect_digit ${project_SRCS})
target_link_libraries(detect_digit ${OpenCV_LIBRARIES})
