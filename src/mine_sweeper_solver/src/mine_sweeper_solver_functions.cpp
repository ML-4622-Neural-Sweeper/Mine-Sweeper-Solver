#include "mine_sweeper_solver_functions.hpp"
#include "mine_sweeper_solver_probs.hpp"
#include "util.hpp"
#include "log.hpp"

namespace slvr
{

void useActionArrays(ActionArray& outClicks, ActionArray& outFlags, mswp::MineSweeper& outBoard)
{
    for (mswp::BoardIndex i = 0; i < outClicks.size(); i++)
    {
        outBoard.click(outClicks[i]);
    }
    outClicks.reset();
    for (mswp::BoardIndex i = 0; i < outFlags.size(); i++)
    {
        outBoard.flag(outFlags[i]);
    }
    outFlags.reset();
}

void lazySolve(MineSweeperSolver& outSolver, ActionArray& outClicks, ActionArray& outFlags)
{
    BoardBitMap isModified;

    outSolver.applyFuncToTilesWithAdjBombs(
    [&outSolver, &outClicks, &outFlags, &isModified](const mswp::BoardIndex i, Tile& outCenterTile) 
    {
        auto width = outSolver.width();
        auto size = outSolver.size();
        auto& tiles = outSolver.tiles();

        /* Flags all adj hidden tiles if adjBombs and adjHidden are equal.
         * 
         * Example:
         * 
         * # # #
         * 2 3 2
         * 
         * '3' has 3 adjHidden tiles and 3 adjBombs.
         * Both 2's have 2 adjHidden tiles and 2 adjBombs.
         *   
         * @ @ @
         * 2 3 2
         * 
         * This allows lazySolve to flag those tiles.
         */
        if (outCenterTile.adjBombs == outCenterTile.adjUnknowns)
        {
            util::applyFuncToAdjObjects<Tiles, Tile>(i, width, size, tiles, 
            [&outFlags, &isModified](int32_t i, Tile& outAdjTile) 
            {
                if (outAdjTile.hidden() && !isModified[i] && outFlags.size() < SLVR_ACTION_BUFFER_SIZE)
                {
                    isModified[i] = true;
                    outFlags.push(i);
                }
            });
        }
        /**
         * Clicks on all hidden tiles if number of adjBombs is zero
         * 
         * Example:
         * 
         * # # 1
         * @ 2 @
         * 
         * The '1' and '2' both effectively become zero due to flagged tiles.
         * 
         * 1 2 1
         * @ 2 @
         * 
         * This allows lazySolve to click the hidden tiles!
         * 
         */
        else if (outCenterTile.adjBombs == 0)
        {
            util::applyFuncToAdjObjects<Tiles, Tile>(i, width, size, tiles, 
            [&outClicks, &isModified](int32_t i, Tile& outAdjTile) 
            {
                if (outAdjTile.hidden() && !isModified[i] && outClicks.size() < SLVR_ACTION_BUFFER_SIZE)
                {
                    outClicks.push(i);
                    isModified[i] = true;
                }
            });
        }
    });
}

struct IntersectionAndDifference
{
    IntersectionAndDifference() : intersectionCount{0}, differenceCenterCount{0}, differenceAdjCount{0} 
    {
        std::fill(intersection.begin(), intersection.end(), -1);
        std::fill(differenceCenter.begin(), differenceCenter.end(), -1);
        std::fill(differenceAdj.begin(), differenceAdj.end(), -1);
    }

    std::array<mswp::BoardIndex, 8> intersection;
    std::array<mswp::BoardIndex, 8> differenceCenter;
    std::array<mswp::BoardIndex, 8> differenceAdj;

    uint8_t intersectionCount;
    uint8_t differenceCenterCount;
    uint8_t differenceAdjCount;
};

IntersectionAndDifference getIntersectionAndDifference(const BoardBitMap& boardBitMapCenter, const BoardBitMap& boardBitMapAdj, 
                                                       const MineSweeperSolver& solver, mswp::BoardIndex centerIndex, mswp::BoardIndex adjIndex)
{
    IntersectionAndDifference interAndDiff;
    util::applyFuncToAdjObjects<Tiles, Tile>(adjIndex, solver.width(), solver.size(), solver.tiles(),
    [&boardBitMapCenter, &interAndDiff](int32_t index, const Tile& tile)
    {
        if (tile.hidden())
        {
            // Intersection
            if (boardBitMapCenter[index] && interAndDiff.intersectionCount < 8)
            {
                interAndDiff.intersection[interAndDiff.intersectionCount] = index;
                interAndDiff.intersectionCount++;
            }
            // Adj Difference
            else if (interAndDiff.differenceAdjCount < 8)
            {
                interAndDiff.differenceAdj[interAndDiff.differenceAdjCount] = index;
                interAndDiff.differenceAdjCount++;
            }
        }
    });

    util::applyFuncToAdjObjects<Tiles, Tile>(centerIndex, solver.width(), solver.size(), solver.tiles(),
    [&boardBitMapAdj, &interAndDiff](int32_t index, const Tile& tile)
    {
        // Center Difference
        if (!boardBitMapAdj[index] && tile.hidden() && interAndDiff.differenceCenterCount < 8)
        {
            interAndDiff.differenceCenter[interAndDiff.differenceCenterCount] = index;
            interAndDiff.differenceCenterCount++;
        }
    });

    return interAndDiff;
}

void applyActionIntersection(uint8_t count, const std::array<mswp::BoardIndex, 8>& indicies, ActionArray& outAction, BoardBitMap& outClickedOrFlagged)
{
    for (int8_t i = 0; i < count; i++)
    {
        if (outClickedOrFlagged[indicies[i]] && outAction.size() < SLVR_ACTION_BUFFER_SIZE)
        {
            continue;
        }
        outClickedOrFlagged[indicies[i]] = true;
        outAction.push(indicies[i]);
    }
}

void intersectionSolver(MineSweeperSolver& outSolver, ActionArray& outClicks, ActionArray& outFlags)
{
    outClicks.reset();
    outFlags.reset();
    BoardBitMap clickedOrFlagged;
    outSolver.applyFuncToTilesWithAdjBombs(
    [&](const mswp::BoardIndex centerIndex, Tile& centerTile) 
    {
        if (centerTile.hidden())
        {
            return;
        }
        auto width = outSolver.width();
        auto size = outSolver.size();
        auto& tiles = outSolver.tiles();
        BoardBitMap presentInCenterTile;
        util::applyFuncToAdjObjects<Tiles, Tile>(centerIndex, outSolver.width(), outSolver.size(), outSolver.tiles(),
        [&presentInCenterTile](int32_t index, Tile& tile)
        {
            presentInCenterTile[index] = true;
        });
        /**
         * 
         * # 3 1 #
         * # # # #
         * 
         * By splitting the board up based off visible tile intersection we can solve more complex puzzles.
         * 
         * A 3 1 B
         * A $ $ B
         * 
         * Since 3 only has two A's we know that 1 must be in $. 
         * Because of this we know that B is free to click. 
         * We also know that A's can be flagged since its restricted to that amount. 
         * 
         */
        util::applyFuncToAdjObjects<Tiles, Tile>(centerIndex, width, size, tiles, 
        [&](int32_t adjIndex, Tile& adjTile) 
        {
            if (adjTile.hidden())
            {
                return;
            }
            BoardBitMap presentInAdjTile;

            util::applyFuncToAdjObjects<Tiles, Tile>(adjIndex, outSolver.width(), outSolver.size(), outSolver.tiles(),
            [&presentInAdjTile](int32_t index, Tile& tile)
            {
                presentInAdjTile[index] = true;
            });

            auto interAndDiff = getIntersectionAndDifference(presentInCenterTile, presentInAdjTile, outSolver, centerIndex, adjIndex);

            int8_t minAmountOfBombsCenter = std::max(centerTile.adjBombs - interAndDiff.intersectionCount, 0);

            int8_t maxAmountOfBombsCenter = std::min(static_cast<int>(centerTile.adjBombs), static_cast<int>(interAndDiff.differenceCenterCount));

            int8_t minAmountOfBombsInter = std::max(centerTile.adjBombs - maxAmountOfBombsCenter, centerTile.adjBombs - maxAmountOfBombsCenter);
            int8_t maxAmountOfBombsInter = std::min(centerTile.adjBombs - minAmountOfBombsCenter, centerTile.adjBombs - minAmountOfBombsCenter);

            int8_t maxBombsInDiffCenter = centerTile.adjBombs - minAmountOfBombsInter;
            int8_t maxBombsInDiffAdj = adjTile.adjBombs - minAmountOfBombsInter;

            int8_t minBombsInDiffCenter = centerTile.adjBombs - maxAmountOfBombsInter;
            int8_t minBombsInDiffAdj = adjTile.adjBombs - maxAmountOfBombsInter;

            // Click entire intersection
            if (maxAmountOfBombsInter == 0 && interAndDiff.intersectionCount != 0)
            {
                applyActionIntersection(interAndDiff.intersectionCount, interAndDiff.intersection, outClicks, clickedOrFlagged);
            }
            // Flag entire intersection
            else if (minAmountOfBombsInter == interAndDiff.intersectionCount && interAndDiff.intersectionCount != 0)
            {
                applyActionIntersection(interAndDiff.intersectionCount, interAndDiff.intersection, outFlags, clickedOrFlagged);
            }

            // Click entire center difference
            if (maxBombsInDiffCenter == 0 && interAndDiff.differenceCenterCount != 0)
            {
                applyActionIntersection(interAndDiff.differenceCenterCount, interAndDiff.differenceCenter, outClicks, clickedOrFlagged);
            }
            // Flag entire center difference
            else if (minBombsInDiffCenter == interAndDiff.differenceCenterCount && interAndDiff.differenceCenterCount != 0)
            {
                applyActionIntersection(interAndDiff.differenceCenterCount, interAndDiff.differenceCenter, outFlags, clickedOrFlagged);
            }

            // Click entire adj difference
            if (maxBombsInDiffAdj == 0 && interAndDiff.differenceAdjCount != 0)
            {
                applyActionIntersection(interAndDiff.differenceAdjCount, interAndDiff.differenceAdj, outClicks, clickedOrFlagged);
            }
            // Flag entire adj difference
            else if (minBombsInDiffAdj == interAndDiff.differenceAdjCount && interAndDiff.differenceAdjCount != 0)
            {
                applyActionIntersection(interAndDiff.differenceAdjCount, interAndDiff.differenceAdj, outFlags, clickedOrFlagged);
            }
        });
    });
}

void getRecommendedActions(MineSweeperSolver& outSolver, ActionArray& outClicks, ActionArray& outFlags)
{
    // Was initially included to improve performance but it can cause clickable tiles top 'disapear' temporarily. 
    // Leads to worse game feel.
    //
    // lazySolve(outSolver, outClicks, outFlags);
    // if (outClicks.size() != 0 || outFlags.size() != 0)
    // {
    //     return;
    // }
    intersectionSolver(outSolver, outClicks, outFlags);
    if (outClicks.size() != 0 || outFlags.size() != 0)
    {
        return;
    }
    TileProbs tileProbs;
    calculateProbs(outSolver, tileProbs);
    mswp::BoardIndex lowestProb = std::min_element(tileProbs.begin(), tileProbs.begin() + outSolver.size()) - tileProbs.begin();
    if (lowestProb >= outSolver.size())
    {
        throw std::runtime_error("Invalid lowestProb!");
    }
}

float getReward(
    mswp::BoardIndex i, float rewardPerTileClicked, float rewardForWinning, 
    float rewardForLosing, float rewardForClickingVisTile, float rewardForClickingFlag, 
    mswp::MineSweeper& outBoard, MineSweeperSolver& outSolver)
{
    ActionArray clicks;
    ActionArray flags;

    mswp::BoardSize totalTilesClicked = 0;

    if (outBoard[i].state == mswp::Tile::VISIBLE)
    {
        return rewardForClickingVisTile;
    }
    if (outBoard[i].state == mswp::Tile::FLAGGED)
    {
        return rewardForClickingFlag;
    }

    outBoard.click(i);
    outSolver.update(outBoard.tileString());
    
    while (true)
    {
        intersectionSolver(outSolver, clicks, flags);
    
        if (outBoard.gameState() == mswp::MineSweeper::WON)
        {
            return rewardForWinning + rewardPerTileClicked*totalTilesClicked;
        }
        
        if (outBoard.gameState() == mswp::MineSweeper::LOST)
        {
            return rewardForLosing + rewardPerTileClicked*totalTilesClicked;
        }
        
        if (clicks.size() == 0 and flags.size() == 0)
        {
            return rewardPerTileClicked*totalTilesClicked;
        }

        mswp::BoardSize tiles_pre_click = outBoard.remainingTile();
        useActionArrays(clicks, flags, outBoard);
        outSolver.update(outBoard.tileString());
        totalTilesClicked += tiles_pre_click - outBoard.remainingTile();
    }
}

};