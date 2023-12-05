project(mSAM-mnn)

cmake_minimum_required(VERSION 3.14.7)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-3.4.18-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc highgui)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20230517-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

# mnn
include_directories(${CMAKE_SOURCE_DIR}/mnn/include/)
#include_directories(${CMAKE_SOURCE_DIR}/mnn/include/MNN/expr/)
include_directories(${CMAKE_SOURCE_DIR}/mnn/cv/include/)
add_library(libMNN STATIC IMPORTED)
add_library(libMNN_CL STATIC IMPORTED)
add_library(libMNN_Express STATIC IMPORTED)
add_library(libMNNOpenCV STATIC IMPORTED)
set_target_properties(
        libMNN
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/mnn/${ANDROID_ABI}/libMNN.so
)
set_target_properties(
        libMNN_Express
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/mnn/${ANDROID_ABI}/libMNN_Express.so
)
set_target_properties(
        libMNN_CL
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/mnn/${ANDROID_ABI}/libMNN_CL.so
)
set_target_properties(
        libMNNOpenCV
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/mnn/${ANDROID_ABI}/libMNNOpenCV.so
)

add_library(mSAM_mnn SHARED jni_intereface.cpp sam.cpp ndkcamera.cpp)

target_link_libraries(mSAM_mnn ncnn ${OpenCV_LIBS} camera2ndk mediandk libMNN libMNN_CL libMNN_Express libMNNOpenCV)
# ncnn: ndkcamera.cpp
