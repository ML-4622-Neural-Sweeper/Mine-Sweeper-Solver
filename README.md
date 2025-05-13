# Neural Sweeper

## How To Compile Binding

NOTE: Compilation requires that you have [CMake](https://cmake.org/download/), a [C++ Compiler](https://isocpp.org/get-started) and a version of [Python](https://www.python.org/downloads/) greater than or equal to 3.8 installed. 

```bash
    cd /path/to/project/
    mkdir -p build && cd build 
    cmake ..
    make
```

## How to Train Model

First modify the hyperparameters before training, the hyper parameters are found in ```Neural-Sweeper/neural_sweeper/src/hyperparameters.py```.

```bash
    cd /path/to/project/
    cd neural_sweeper
    python3 train.py
```

NOTE: If you receive an error when running ```python3 train.py``` make sure you used the same version of python that cmake during compiling of the bindings! CMake will print out the version that its uses when it compiles.

## How to Test Model

```bash
    cd /path/to/project/
    cd neural_sweeper
    python3 test.py
```

## Packages Used
- [Google Benchmark](https://github.com/google/benchmark)
- [Google Test](https://github.com/google/googletest)
- [Cpp-Terminal](https://github.com/jupyter-xeus/cpp-terminal)