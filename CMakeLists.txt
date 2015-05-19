# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 2.8)
project(mpp)

# guard against in-source builds (got this from Eigen)
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif()

#add the customised package searches to the module path
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_tolower)
if(NOT cmake_build_type_tolower STREQUAL "debug" AND NOT cmake_build_type_tolower STREQUAL "release")
    message(FATAL_ERROR "Unknown build type \"${CMAKE_BUILD_TYPE}\". Allowed values are Debug and Release.")
endif()

# C and C++ compiler flags
include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)

macro(mpp_add_cxx_compiler_flag FLAG)
    string(REGEX REPLACE "-" "" SFLAG1 ${FLAG})
    string(REGEX REPLACE "\\+" "p" SFLAG ${SFLAG1})
    check_cxx_compiler_flag(${FLAG} COMPILER_SUPPORT_${SFLAG})
    if(COMPILER_SUPPORT_${SFLAG})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAG}")
    endif()
endmacro(mpp_add_cxx_compiler_flag)

macro(mpp_add_c_compiler_flag FLAG)
    string(REGEX REPLACE "-" "" SFLAG1 ${FLAG})
    string(REGEX REPLACE "\\+" "p" SFLAG ${SFLAG1})
    check_c_compiler_flag(${FLAG} COMPILER_SUPPORT_${SFLAG})
    if(COMPILER_SUPPORT_${SFLAG})
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${FLAG}")
    endif()
endmacro(mpp_add_c_compiler_flag)

if(NOT MSVC)
    # We assume that other compilers are partly compatible with GNUCC
    set(CMAKE_CXX_FLAGS_DEBUG "-g3")
    set(CMAKE_CXX_FLAGS_RELEASE "-g0 -O3")

    set(CMAKE_C_FLAGS_DEBUG "-g3")
    set(CMAKE_C_FLAGS_RELEASE "-g0 -O3")

    mpp_add_cxx_compiler_flag("-pedantic")
    mpp_add_cxx_compiler_flag("-Wall")
    mpp_add_cxx_compiler_flag("-Wextra")
    mpp_add_cxx_compiler_flag("-Wfatal-errors")
    mpp_add_cxx_compiler_flag("-std=c++11")
    mpp_add_cxx_compiler_flag("-fPIC")

    mpp_add_c_compiler_flag("-pedantic")
    mpp_add_c_compiler_flag("-Wall")
    mpp_add_c_compiler_flag("-Wextra")
    mpp_add_c_compiler_flag("-Wfatal-errors")
    mpp_add_c_compiler_flag("-std=c11")
    mpp_add_c_compiler_flag("-fPIC")
endif(NOT MSVC)

find_package(Eigen3 3.1.2 REQUIRED)
find_package(Boost REQUIRED COMPONENTS unit_test_framework serialization filesystem system log thread date_time)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR})

if(MPP_ENABLE_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif(MPP_ENABLE_TESTING)

add_subdirectory(libmpp)
add_subdirectory(examples)
add_subdirectory(pympp)
