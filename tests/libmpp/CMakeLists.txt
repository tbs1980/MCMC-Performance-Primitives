# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 2.8)
project(TestUtils C)

include_directories(${CMAKE_SOURCE_DIR}/libmpp/include)

file(GLOB test_SRCS "*.c")

foreach(test_src ${test_SRCS})
    get_filename_component(test ${test_src} NAME_WE)
    add_executable(test_${test} ${test_src})
    target_link_libraries(test_${test} mpp ${Boost_LIBRARIES} stdc++)
    add_test(libmpp_${test} test_${test})
endforeach(test_src)
