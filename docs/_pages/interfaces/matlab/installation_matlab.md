---
title: Installation (Matlab)
layout: default
parent: Matlab / Octave
nav_order: 1
---

## Installing precompiled package

PIQP can be directly installed running the following commands

```matlab
websave('install_piqp.m','https://raw.githubusercontent.com/PREDICT-EPFL/piqp/main/interfaces/matlab/install_piqp.m');
install_piqp
```

## Building and Installing from Source

{% root_include _common/building_from_source_deps.md %}

### Building and Installing PIQP

* Clone PIQP from Github
```shell
git clone https://github.com/PREDICT-EPFL/piqp.git
```
* Build the interface in Matlab by executing the following commands
```matlab
cd interfaces/matlab
% to build with Blasfeo (needed for sparse_multistage backend)
% setenv("CMAKE_ARGS", "-DBUILD_WITH_BLASFEO=ON")
make_piqp
```
This will build and package the Matlab interface into a `piqp-matlab-{platform}64.tar.gz` file.
You can also directly add the interface to the search path in Matlab by running
```matlab
addpath(pwd)
savepath
```
