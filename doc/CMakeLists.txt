cmake_minimum_required(VERSION 2.8)

find_package(Doxygen)

if(DOXYGEN_FOUND)
    set (doxygenConfigFileIn "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in")
    set (doxygenConfigFile "${CMAKE_CURRENT_BINARY_DIR}/mpp.dox")
    configure_file(${doxygenConfigFileIn} ${doxygenConfigFile} @ONLY)
    add_custom_target (MPPDocs
        ${DOXYGEN_EXECUTABLE} ${doxygenConfigFile}
        DEPENDS ${doxygenConfigFile}
    VERBATIM)
endif(DOXYGEN_FOUND)
