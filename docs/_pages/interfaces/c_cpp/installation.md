---
title: Installation 
layout: default
parent: C/C++
nav_order: 1
---

## Installing using conda

PIQP can be directly installed via anaconda/miniconda:

```shell
conda install -c conda-forge piqp
```

## Building and Installing from Source

{% root_include _common/building_from_source_deps.md %}

### Building and Installing PIQP

* Clone PIQP from Github
```shell
git clone https://github.com/PREDICT-EPFL/piqp.git
```
* Build PIQP in a `build` folder
```shell
cd piqp
mkdir build
cd build
# add -DBUILD_WITH_BLASFEO=ON to build with Blasfeo (needed for sparse_multistage backend)
cmake .. -DCMAKE_CXX_FLAGS="-march=native" -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF
cmake --build . --config Release
```
Note that by setting `-march=native`, we allow the compiler to optimize for the full available instruction set on the machine compiling the code.

{: .warning }
When compiling with `-march=native` and you are on a modern x86 architecture (which you are very very likely are), Eigen will align vectors to 32 (AVX2) or 64 (AVX512) bytes. Hence, when consuming the precompiled PIQP library, your target needs to be built with the same architecture flags (e.g. `-march=native`) otherwise there will be ABI incompatibilities with Eigen. Alternatively, by setting the CMake flag `-DBUILD_WITH_EIGEN_MAX_ALIGN_BYTES=ON`, PIQP will be built with `EIGEN_MAX_ALIGN_BYTES=64` ensuring maximal Eigen compatibility and export it as well to the linked target. Note that this might conflict with other libraries and has to be used cautiously. For more information see the *Alignment* section in the [Eigen docs](https://eigen.tuxfamily.org/dox/TopicPreprocessorDirectives.html).

* Install libraries and header files (requires CMake 3.15+)
```shell
cmake --install . --config Release
```
This will install the C++ and C headers and shared libraries.

{: .note }
If you want to build static libraries instead, you can pass `-DBUILD_SHARED_LIBS=OFF` when configuring cmake.

## Using PIQP in CMake Projects

PIQP has first class support for CMake project. The C++ library is header-only. For the C interface we provide a shared as well as a static library which can be linked against.

```cmake
# Find PIQP package
find_package(piqp REQUIRED)

# PIQP requires at least C++14
set(CMAKE_CXX_STANDARD 14)

# Link the PIQP C++ library with precompiled template instantiations
target_link_libraries(yourTarget PRIVATE piqp::piqp)

# Link the PIQP C++ header-only library
target_link_libraries(yourTarget PRIVATE piqp::piqp_header_only)

# Link the PIQP C library
target_link_libraries(yourTarget PRIVATE piqp::piqp_c)
```
