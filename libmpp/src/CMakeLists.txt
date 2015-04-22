cmake_minimum_required(VERSION 2.8)

add_library(mpp_static STATIC mpp_hmc.cpp)
add_library(mpp SHARED mpp_hmc.cpp)

# http://www.cmake.org/pipermail/cmake/2010-July/037892.html

install(TARGETS mpp_static
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    COMPONENT Devel)

install(TARGETS mpp
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    COMPONENT Devel)
