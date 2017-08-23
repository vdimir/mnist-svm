cmake_minimum_required(VERSION 3.7)
project(task)

set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES src/main.cpp src/InputFileEntry.h src/Svm.cpp src/Svm.h src/common.h src/FeatureExtractor.h)

find_library(VLFEAT_LIBRARY NAMES vl libvl)
find_package(OpenCV REQUIRED)

add_executable(task ${SOURCE_FILES})
target_link_libraries(task ${VLFEAT_LIBRARY} ${OpenCV_LIBS})
