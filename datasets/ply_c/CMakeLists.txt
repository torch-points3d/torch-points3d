# Graph for Cut Pursuit
# author: Loic Landrieu
# date: 2017

CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

PROJECT(LIBGEOF)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -fopenmp -O3")

##############################
### Find required packages ###
##############################
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

find_package(PythonLibs)
find_package(PythonInterp)
find_package(NumPy 1.5 REQUIRED)

find_package(Boost 1.65.0 COMPONENTS graph REQUIRED) #system filesystem thread serialization
if (${Boost_MINOR_VERSION} LESS 67 )
    find_package(Boost 1.65.0 COMPONENTS numpy${PYTHON_VERSION_MAJOR} REQUIRED) #system filesystem thread serialization
else()
    set(PYTHONVERSION ${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})
    find_package(Boost 1.67.0 COMPONENTS numpy${PYTHONVERSION} REQUIRED)
endif()

include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

message("Boost includes ARE " ${Boost_INCLUDE_DIRS})
message("Boost LIBRARIES ARE " ${Boost_LIBRARY_DIRS})

find_package(Eigen3 REQUIRED NO_MODULE)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})
#LINK_DIRECTORIES(${EIGEN3_LIBRARY_DIRS})

#SET(PYTHON_LIBRARIES /usr/lib/x86_64-linux-gnu/libpython2.7.so)
#SET(PYTHON_INCLUDE_DIRS /usr/include/python2.7)

message("PYTHON LIBRARIES ARE " ${PYTHON_LIBRARIES})
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS} ${PYTHON_NUMPY_INCLUDE_DIR})
LINK_DIRECTORIES(${PYTHON_LIBRARY_DIRS})
##############################
###  Build target library  ###
##############################

set(CMAKE_LD_FLAG "${CMAKE_LD_FLAGS} -shared -Wl -fPIC --export-dynamic -fopenmp -O3 -Wall")

add_library(ply_c SHARED ply_c.cpp)
target_link_libraries(ply_c
    ${Boost_LIBRARIES}
    ${PYTHON_LIBRARIES}
    )
