# Copyright 2025 Dalton Prokosch

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCE
    src/app.cpp
)

set(CPPTERMINAL_BUILD_EXAMPLES OFF)
set(CPPTERMINAL_ENABLE_TESTING OFF)
set(CPPTERMINAL_ENABLE_DOCS OFF)


include(FetchContent)
FetchContent_Declare(cpp-terminal
    GIT_REPOSITORY https://github.com/jupyter-xeus/cpp-terminal
    GIT_TAG f1d2e1c)

FetchContent_MakeAvailable(
    cpp-terminal)

add_library(app STATIC ${SOURCE})
target_include_directories(app PUBLIC include)
target_link_libraries(app mine_sweeper mine_sweeper_solver util cpp-terminal)
