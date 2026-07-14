cmake_minimum_required(VERSION 3.16.3)
project(sitk_bench_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Reference ITK build (ITK 6.0, static). Override with -DITK_DIR=...
if(NOT ITK_DIR)
  set(ITK_DIR "/home/stevek/work/ITK-worktrees/verify-build" CACHE PATH "" FORCE)
endif()

find_package(ITK REQUIRED
  COMPONENTS
    ITKCommon
    ITKImageIntensity
    ITKSmoothing
    ITKImageGradient
    ITKImageFilterBase
    ITKBinaryMathematicalMorphology
    ITKMathematicalMorphology
    ITKDistanceMap
    ITKConnectedComponents
    ITKThresholding
    ITKConvolution
    ITKFFT
)
include(${ITK_USE_FILE})

add_executable(sitk_bench_cpp main.cxx)
target_link_libraries(sitk_bench_cpp PRIVATE ${ITK_LIBRARIES})
