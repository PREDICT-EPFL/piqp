---
title: Installation (Octave)
layout: default
parent: Matlab / Octave
nav_order: 2
---

## Installing package

{: .warning }
The Octave interface doesn't support the KKT solver backend `sparse_multistage`.

PIQP can be directly installed running the following command

```matlab
pkg install "https://github.com/PREDICT-EPFL/piqp/releases/latest/download/piqp-octave.tar.gz"
```

Before using PIQP in Octave make sure to load it in your script with
```matlab
pkg load piqp
```

## Building and Installing from Source

* Clone PIQP from Github
```shell
git clone https://github.com/PREDICT-EPFL/piqp.git
```
* Package PIQP in Octave by executing the following commands
```matlab
cd interfaces/octave
package_piqp
```
This will download all dependencies and package the Octave interface into a `piqp-octave.tar.gz` file.
You can then install the package with
```matlab
pkg install piqp-octave.tar.gz
```
