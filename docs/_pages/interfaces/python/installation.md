---
title: Installation 
layout: default
parent: Python
nav_order: 1
---

## Installing using pip

PIQP can be directly installed via pip:

```shell
pip install piqp
```

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
* Building and installing PIQP
```shell
cd piqp
# to build with Blasfeo (needed for sparse_multistage backend)
# export CMAKE_ARGS="-DBUILD_WITH_BLASFEO=ON"
python3 -m pip install .
```
This will build and install piqp. Alternatively, also a wheel can be build using
```shell
python3 -m build --wheel
```
and then installed using
```shell
python3 -m pip install dist/<build-wheel>.whl
```
