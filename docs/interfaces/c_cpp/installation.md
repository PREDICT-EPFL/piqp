---
title: Installation 
layout: default
parent: C/C++
grand_parent: Interfaces
nav_order: 1
---

## Building and Installing from Source

{% root_include _common/building_from_source_deps.md %}

### Building and Installing PIQP

* Clone PIQP from Github recursively
```shell
git clone https://github.com/PREDICT-EPFL/piqp.git --recurse-submodules
```
* Build PIQP in a `build` folder
```shell
cd piqp
mkdir build
cd build
cmake .. -DCMAKE_CXX_FLAGS="-march=native" -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF
cmake --build .
```
Note that by setting `-march=native`, we allow the compiler to optimize for the full available instruction set on the machine compiling the code.
* Install libraries and header files (requires CMake 3.15+)
```shell
cmake --install .
```
This will install the C++ and C headers and the static and shared library for the C interface.

## Using PIQP in CMake Projects

PIQP has first class support for CMake project. The C++ library is header-only. For the C interface we provide a shared as well as a static library which can be linked against.

```cmake
# Find PIQP package
find_package(piqp REQUIRED)

# PIQP requires at least C++14
set(CMAKE_CXX_STANDARD 14)

# Link the PIQP C++ header-only library
target_link_libraries(yourTarget PRIVATE piqp::piqp)

# Link the PIQP C shared library
target_link_libraries(yourTarget PRIVATE piqp::piqp_c_shared)
# or link the PIQP C static library
target_link_libraries(yourTarget PRIVATE piqp::piqp_c_static)
```
