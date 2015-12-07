cmake_minimum_required(VERSION 2.8)

project(examples_lbmpp C)

include_directories(${CMAKE_SOURCE_DIR}/libmpp/include)

file(GLOB example_lbmpp_SRCS "*.c")

foreach(example_lbmpp_src ${example_lbmpp_SRCS})
    get_filename_component(example_lbmpp ${example_lbmpp_src} NAME_WE)
    add_executable(example_lbmpp_${example_lbmpp} ${example_lbmpp_src})
    target_link_libraries(example_lbmpp_${example_lbmpp} mpp_static ${Boost_LIBRARIES} stdc++ ${CMAKE_THREAD_LIBS_INIT})
    add_test(lbmpp_${example_lbmpp} example_lbmpp_${example_lbmpp})
    install(TARGETS example_lbmpp_${example_lbmpp}
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        COMPONENT Devel)
endforeach(example_lbmpp_src)
