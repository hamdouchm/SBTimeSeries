# Minimum cmake version
cmake_minimum_required(VERSION 3.11)


# Put project or library name
project(SBTimeSeries LANGUAGES CXX)

# define source files
file(GLOB SOURCE "src/*.cpp" "src/*.h")

# find pybind11
set(pybind11_DIR ~/.include/share/cmake/pybind11)
find_package(pybind11 REQUIRED)
pybind11_add_module(SBTimeSeries ${SOURCE})

# build option
option(CMAKE_BUILD_TYPE Release)