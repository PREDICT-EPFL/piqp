---
title: Installation 
layout: default
parent: Octave
grand_parent: Interfaces
nav_order: 1
---

## Building from Source (on Linux)

The PIQP Octave interface currently can only be built from source.  To start with,
follow the instructions to [install Eigen and prerequistes](../../_common/building_from_source_deps.md)

Then, from the piqp root directory do
```
$ cmake -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_OCTAVE_INTERFACE=ON .
$ make piqp_plain.oct
```

Then you can start Octave with
```
$ LD_PRELOAD=[path-to-piqp]/interfaces/c/libpiqpc.so octave
```

When Octave starts, you also need to execute in Octave:
```
octave:1> autoload("__piqp__", "[path-to-piqp]/interfaces/octave/piqp_plain.oct");
octave:2> addpath("[path-to-piqp]/interfaces/octave");
```

## Installing

TODO - implement an install step in Cmake to put `libpiqpc.so` in a known shared library directory so the LD_PRELOAD isn't necessary.  And also put the `piqp_plain.oct` and `piqp.m` files in areas for Octave to find.

## Running

In Octave you can now
```
octave:3> help piqp
```
