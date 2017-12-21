project(ProbmapGenerator)

SET(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "..." FORCE)

# Define headers for this library. PUBLIC headers are used for
# compiling the library, and will be added to consumers' build
# paths.
add_executable(generator generate_probmap.cpp WsiPredictor.cpp cnpy.cpp)
target_include_directories(generator PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE *.cpp)


target_link_libraries(generator config4cpp ${OpenCV_LIBS} ${Caffe_LIBRARIES} ${OPENSLIDE_LIBRARY}) 

# 'make install' to the correct locations (provided by GNUInstallDirs).
install(TARGETS generator EXPORT ProbmapGeneratorConfig
    ARCHIVE  DESTINATION file
    LIBRARY  DESTINATION lib
    RUNTIME  DESTINATION bin)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)

#INSTALL(DIRECTORY ${Caffe_DIR}/../../lib/ DESTINATION bin)
INSTALL(DIRECTORY ${Caffe_DIR}/lib/ DESTINATION bin)

# This makes the project importable from the install directory
# Put config file in per-project dir (name MUST match), can also
# just go into 'cmake'.
install(EXPORT ProbmapGeneratorConfig DESTINATION share/ProbmapGenerator/cmake)

# This makes the project importable from the build directory
export(TARGETS generator FILE ProbmapGeneratorConfig.cmake)

