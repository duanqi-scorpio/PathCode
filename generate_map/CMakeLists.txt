project(CAMELYON16)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "..." FORCE)

include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

# link to config4cpp
ExternalProject_Add(project_config4cpp
  URL http://www.config4star.org/download/config4cpp.tar.gz
  SOURCE_DIR ${CMAKE_BINARY_DIR}/config4cpp-src
  BINARY_DIR ${CMAKE_BINARY_DIR}/config4cpp-build
  INSTALL_DIR ${CMAKE_BINARY_DIR}/config4cpp-src
  #PREFIX ${CMAKE_CURRENT_BINARY_DIR}/config4cpp
  CONFIGURE_COMMAND ""
  BUILD_COMMAND cd ${CMAKE_BINARY_DIR}/config4cpp-src && make
  INSTALL_COMMAND ""
  #PREFIX=${CMAKE_CURRENT_BINARY_DIR}/config4cpp
)

ExternalProject_Get_Property(project_config4cpp install_dir)
add_library(config4cpp STATIC IMPORTED)
set_property(TARGET config4cpp PROPERTY IMPORTED_LOCATION ${install_dir}/lib/libconfig4cpp.a)
add_dependencies(config4cpp project_config4cpp)
include_directories(${install_dir}/include)

# link to openslide lib
include_directories(/usr/include/openslide)
find_library(OPENSLIDE_LIBRARY openslide HINTS /usr/lib/x86_64-linux-gnu)

# link to OpenCV
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

# link to Caffe
find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY

add_subdirectory(src)


