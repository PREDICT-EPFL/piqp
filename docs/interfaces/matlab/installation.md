---
title: Installation 
layout: default
parent: Matlab
grand_parent: Interfaces
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

* Clone PIQP from Github recursively
```shell
git clone https://github.com/PREDICT-EPFL/piqp.git --recurse-submodules
```
* Build the interface in Matlab by executing the following commands
```matlab
cd interfaces/matlab
make_piqp
```
This will build and package the Matlab interface into a `piqp-matlab-{platform}64.tar.gz` file.
You can also directly add the interface to the search path in Matlab by running
```matlab
addpath(pwd)
savepath
```
