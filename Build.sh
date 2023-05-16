#!/bin/bash

# build pybind11
mkdir -p extern/pybind11/build
cd extern/pybind11/build
mkdir -p ~/.include
cmake .. -DCMAKE_INSTALL_PREFIX=~/.include
make install

# build solution
cd ../../..
mkdir -p build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/.include
make
cd ..