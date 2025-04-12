#pragma once

#include "mine_sweeper_solver.hpp"
#include <algorithm>

namespace slvr
{

typedef std::array<mswp::BoardIndex, SLVR_ACTION_BUFFER_SIZE> ActionBuffer;
typedef uint8_t ActionsSize;

class ActionArray
{
public:
    ActionArray() : m_Size{0} {}

    inline void push(mswp::BoardIndex i)
    {
        m_Actions[m_Size] = i;
        m_Size++;
    }

    inline bool in(mswp::BoardIndex item) const
    {
        for (ActionsSize j = 0; j < m_Size; j++)
        {
            if (m_Actions[j] == item)
            {
                return true;
            }
        }
        return false;
    }

    inline void reset()
    {
        m_Size = 0;
    }

    inline mswp::BoardIndex operator[](ActionsSize i) const
    {
        return m_Actions[i];
    }

    inline mswp::BoardIndex& operator[](ActionsSize i)
    {
        return m_Actions[i];
    }

    inline ActionBuffer::iterator begin()
    {
        return m_Actions.begin();
    }

    inline ActionBuffer::iterator end()
    {
        return m_Actions.begin() + m_Size;
    }

    inline ActionBuffer::const_iterator begin() const
    {
        return m_Actions.begin();
    }

    inline ActionBuffer::const_iterator end() const
    {
        return m_Actions.begin() + m_Size;
    }

    inline ActionsSize size() const
    {
        return m_Size;
    }

    inline const ActionBuffer& actions() const
    {
        return m_Actions;
    }

    inline void remove(mswp::BoardIndex item)
    {
        auto newEnd = std::remove(begin(), end(), item);
        m_Size = newEnd - begin();
    }

    inline void update(const mswp::TileString& tileString)
    {
        for (mswp::TileStringIndex i = 0; i < tileString.size(); i++)
        {
            if (tileString[i] != mswp::TileChar::HIDDEN)
            {
                remove(i);
            }
        }
    }

private:
    ActionBuffer m_Actions;
    ActionsSize m_Size;
};

inline std::ostream& operator<<(std::ostream &out, const ActionArray& array)
{
    out << "{ ";
    for (mswp::BoardIndex i : array)
    {
        out << i << ", ";
    }
    out << "}\n";
    return out;
}

void useActionArrays(ActionArray& outClicks, ActionArray& outFlags, mswp::MineSweeper& outBoard);

void lazySolve(MineSweeperSolver& outSolver, ActionArray& outClicks, ActionArray& outFlags);
void intersectionSolver(MineSweeperSolver& outSolver, ActionArray& outClicks, ActionArray& outFlags);

void getRecommendedActions(MineSweeperSolver& outSolver, ActionArray& outClicks, ActionArray& outFlags);

/**
 * @brief Get reward for tiles clicked, used by python in conjunction with ML packages.
 * 
 * @param x x position clicked
 * @param y y position clicked
 * @param rewardPerTileClicked amount multiplied by tiles clicked
 * @param rewardForWinning amount gained when game is won
 * @param rewardForLosing amount gained when game is lost
 * @param outBoard board
 * @param outSolver solver
 * @return float 
 */
float getReward(
    mswp::BoardXPos x, mswp::BoardYPos y, float rewardPerTileClicked, float rewardForWinning, 
    float rewardForLosing, mswp::MineSweeper& outBoard, MineSweeperSolver& outSolver);

}