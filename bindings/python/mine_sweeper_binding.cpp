#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "mine_sweeper.hpp"
#include "mine_sweeper_solver.hpp"
#include "mine_sweeper_solver_functions.hpp"

#include <string>

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(mswp::BoardIndex);

PYBIND11_MODULE(MineSweeper, m)
{
    py::class_<mswp::TileString>(m, "TileString")
        .def(py::init<>());
    py::class_<mswp::MineSweeper>(m, "MineSweeper")
        .def(py::init(
        [](mswp::BoardWidth width, mswp::BoardHeight height, mswp::BombCount bombCount, mswp::BoardSeed seed) 
        {
            return mswp::MineSweeper(width, height, bombCount, seed);
        }))
        .def(py::init(
        [](mswp::BoardWidth width, mswp::BoardHeight height, mswp::BombCount bombCount) 
        {
            return mswp::MineSweeper(width, height, bombCount, time(0));
        }))
        .def("click", static_cast<bool (mswp::MineSweeper::*)(mswp::BoardIndex)>(&mswp::MineSweeper::click))
        .def("click", static_cast<bool (mswp::MineSweeper::*)(mswp::BoardXPos, mswp::BoardYPos)>(&mswp::MineSweeper::click))
        .def("flag", static_cast<bool (mswp::MineSweeper::*)(mswp::BoardIndex)>(&mswp::MineSweeper::flag))
        .def("flag", static_cast<bool (mswp::MineSweeper::*)(mswp::BoardXPos, mswp::BoardYPos)>(&mswp::MineSweeper::flag))
        .def("tile_string", &mswp::MineSweeper::tileString)
        .def("size", &mswp::MineSweeper::size)
        .def("remaining_tiles", &mswp::MineSweeper::remainingTile)
        .def("width", &mswp::MineSweeper::width)
        .def("game_state", &mswp::MineSweeper::gameState)
        .def("flags_remaining", &mswp::MineSweeper::flagsRemaining)
        .def("reset", &mswp::MineSweeper::reset)
        .def("__repr__", [](const mswp::MineSweeper& minesweeper) 
        {
            std::ostringstream stream;
            stream << minesweeper;
            std::string str =  stream.str();
            return str;
        });

    py::enum_<mswp::MineSweeper::GameState>(m, "GameState")
        .value("START", mswp::MineSweeper::GameState::START)
        .value("IN_PROGRESS", mswp::MineSweeper::GameState::IN_PROGRESS)
        .value("LOST", mswp::MineSweeper::GameState::LOST)
        .value("WON", mswp::MineSweeper::GameState::WON);
    
    py::class_<slvr::Tile>(m, "Tile")
        .def(py::init<>())
        .def("hidden", &slvr::Tile::hidden)
        .def("is_bomb", &slvr::Tile::isBomb)
        .def_readwrite("adj_bombs", &slvr::Tile::adjBombs)
        .def_readwrite("adj_hidden", &slvr::Tile::adjUnknowns);

    py::class_<slvr::Tiles>(m, "Tiles")
        .def("__getitem__", [](slvr::Tiles &self, size_t index) {
            return self[index];  // Calls operator[]
        });

    py::class_<slvr::MineSweeperSolver>(m, "MineSweeperSolver")
        .def(py::init<mswp::MineSweeper>())
        .def("update", &slvr::MineSweeperSolver::update)
        .def("width", &slvr::MineSweeperSolver::width)
        .def("size", &slvr::MineSweeperSolver::size)
        .def("deep_tiles_remaining", [](slvr::MineSweeperSolver& solver) 
        {
            return static_cast<int>(solver.remainingDeepTiles());
        })
        .def("tiles", static_cast<const slvr::Tiles& (slvr::MineSweeperSolver::*)() const>(&slvr::MineSweeperSolver::tiles))
        .def("__repr__", [](const slvr::MineSweeperSolver& solver) 
        {
            std::ostringstream stream;
            stream << solver;
            std::string str = stream.str();
            return str;
        });
    
    py::class_<slvr::ActionArray>(m, "ActionArray")
        .def(py::init<>())
        .def("size", &slvr::ActionArray::size)
        .def("push", [](slvr::ActionArray& actionArray, int i) 
        {
            actionArray.push(static_cast<mswp::BoardIndex>(i));
        })
        .def("reset", &slvr::ActionArray::reset);

    m.def("use_action_arrays", slvr::useActionArrays);
    m.def("lazy_solve", slvr::lazySolve);
    m.def("intersection_solve", slvr::intersectionSolver);

    m.def("recommended_actions", slvr::getRecommendedActions);

    m.def("get_reward", [](int index, float r0, float r1, float r2, float r3, float r4, mswp::MineSweeper& board, slvr::MineSweeperSolver& solver) 
    {
        return slvr::getReward(index, r0, r1, r2, r3, r4, board, solver);
    });
}