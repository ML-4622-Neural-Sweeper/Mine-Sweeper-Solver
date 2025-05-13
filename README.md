# Neural Sweeper

Neural Sweeper is a program that trains a Reinforcement Learning (RL) agent to find the best place to click when a guess must be made. The RL model uses a convolutional neural network (CNN) to identify patterns in the game state.

## How to Compile the Python Bindings

NOTE: Compilation requires that you have [CMake](https://cmake.org/download/), a [C++ Compiler](https://isocpp.org/get-started) and a version of [Python](https://www.python.org/downloads/) greater than or equal to 3.8 installed. 

```bash
    cd /path/to/project/
    mkdir -p build && cd build 
    cmake ..
    make
```

## How to Train the Model

First, modify the hyperparameters located in `neural_sweeper/src/hyperparameters.py` before training.

```bash
    cd /path/to/project/
    cd neural_sweeper
    python3 train.py
```

**Note:** If you encounter an error running `python3 train.py`, ensure you're using the same Python version that CMake used when compiling the bindings. CMake will print the Python version it detects during configuration.


## How to Test the Model

```bash
    cd /path/to/project/
    cd neural_sweeper
    python3 test.py
```

## File Structure
```bash
Neural-Sweeper/
├── benchmark/      # Benchmarks for MineSweeper and utilities
├── bindings/       # C++ to Python bindings
├── neural_sweeper/ # Python code for RL training and testing
├── src/            # Core C++ implementation
├── test/           # Unit and integration tests
├── docs/           # Documentation assets
├── CMakeLists.txt
└── README.md
```
## Packages Used
- [Google Benchmark](https://github.com/google/benchmark)
- [Google Test](https://github.com/google/googletest)
- [Cpp-Terminal](https://github.com/jupyter-xeus/cpp-terminal)
