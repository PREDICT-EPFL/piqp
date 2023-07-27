To build PIQP it is required to have [CMake](https://cmake.org/), [Eigen 3.3.4+](https://eigen.tuxfamily.org/index.php?title=Main_Page) and a compatible compiler like [GCC](https://gcc.gnu.org/), [Clang](https://clang.llvm.org/), or [Visual Studio](https://visualstudio.microsoft.com/de/) with C++ extensions on Windows installed. CMake and a compatible compiler should already be installed on most systems.

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
