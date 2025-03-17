To build PIQP it is required to have [CMake](https://cmake.org/), [Eigen 3.3.4+](https://eigen.tuxfamily.org/index.php?title=Main_Page) and a compatible compiler like [GCC](https://gcc.gnu.org/), [Clang](https://clang.llvm.org/), or [Visual Studio](https://visualstudio.microsoft.com/de/) with C++ extensions on Windows installed. CMake and a compatible compiler should already be installed on most systems. The (optional) KKT solver backend `sparse_multistage` needs the additional dependency [Blasfeo](https://github.com/giaf/blasfeo).

### Installing Eigen

#### on macOS via Homebrew

```shell
brew install eigen
```

#### on Ubuntu

```shell
sudo apt install libeigen3-dev
```

#### on Windows via Chocolatey

```shell
choco install eigen
```

#### via conda

```shell
conda install -c conda-forge eigen
```

#### building from source

```shell
# clone Eigen
git clone https://gitlab.com/libeigen/eigen.git eigen
cd eigen
git checkout 3.4.0

# build Eigen
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .

# install Eigen
cmake --install .
```

### Installing Blasfeo (optional)

#### via conda

```shell
conda install -c conda-forge blasfeo
```

{: .note }
Blasfeo installed via conda is a dynamic library and not static. Additionally, it is never build with the `X64_INTEL_SKYLAKE_X` target. This means if you care about static linkage or running an x86-64 CPU with AVX512 support, you might lose some performance, and building from source is recommended.

#### building from source (unix only)

```shell
# clone Blasfeo
git clone https://github.com/giaf/blasfeo.git blasfeo
cd blasfeo

# build Blasfeo
mkdir build
cd build
# -DTARGET=X64_INTEL_SKYLAKE_X    on x86_64 CPUs with AVX512 support (Intel Skylake / AMD Zen5 or later)
# -DTARGET=X64_INTEL_HASWELL      on x86_64 CPUs with AVX2 support (Intel Haswell / AMD Zen or later)
# -DTARGET=X64_INTEL_SANDY_BRIDGE on x86_64 CPUs with AVX support (Intel Sandy-Bridge or later)
# -DTARGET=X64_INTEL_CORE         on x86_64 CPUs with SSE3 support (Intel Core or later)
# -DTARGET=X64_AMD_BULLDOZER      on x86_64 CPUs with AVX and FMA support (AMD Bulldozer or later)
# -DTARGET=ARMV8A_APPLE_M1        on ARMv8A CPUs optimized for Apple M1 or later
# -DTARGET=ARMV8A_ARM_CORTEX_A76  on ARMv8A CPUs optimized for ARM Cortex A76 (e.g. Raspberry Pi 5) 
# -DTARGET=ARMV8A_ARM_CORTEX_A73  on ARMv8A CPUs optimized for ARM Cortex A73
# for more targets see https://github.com/giaf/blasfeo/blob/master/CMakeLists.txt#L55
cmake .. -DCMAKE_BUILD_TYPE=Release -DTARGET=X64_INTEL_HASWELL
cmake --build .

# install Blasfeo
cmake --install .
```