cmake_minimum_required(VERSION 2.8)

project(ExamplesHamiltonian CXX)

file(GLOB example_SRCS "*.cpp")

foreach(example_src ${example_SRCS})
    get_filename_component(example ${example_src} NAME_WE)
    add_executable(example_${example} ${example_src})
    target_link_libraries(example_${example} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
    add_test(Hamiltonian_${example} example_${example})
endforeach(example_src)
