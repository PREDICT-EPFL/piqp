---
title: Installation 
layout: default
parent: Python
grand_parent: Interfaces
nav_order: 1
---

## Installing using pip

PIQP can be directly installed via pip:

```shell
pip install piqp
```

## Building and Installing from Source

{% root_include _common/building_from_source_deps.md %}

### Building and Installing PIQP

* Clone PIQP from Github recursively
```shell
git clone https://github.com/PREDICT-EPFL/piqp.git --recurse-submodules
```
* Building and installing PIQP
```shell
cd piqp
python3 -m pip install .
```
This will build and install pipq. Alternatively, also a wheel can be build using
```shell
python3 -m build --wheel
```
and then installed using
```shell
python3 -m pip install dist/<build-wheel>.whl
```
