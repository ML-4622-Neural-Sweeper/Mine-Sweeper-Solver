#include "mine_sweeper_solver_probs.hpp"
#include "util.hpp"
#include "log.hpp"

namespace slvr
{

    namespace group
    {

    void findHiddenTiles(mswp::BoardIndex centerVisibleTileIndex, const MineSweeperSolver& solver, BoardBitMap& outVisited, TileGroup& outGroup)
    {
        outVisited[centerVisibleTileIndex] = true;
        outGroup.tiles[outGroup.size].tileIndex = centerVisibleTileIndex;
        outGroup.size += 1;
        util::applyFuncToAdjObjects<MineSweeperSolver, Tile>(centerVisibleTileIndex, solver, 
        [&](int32_t i, const Tile& tile)
        {
            if (!tile.hidden() || tile.bombProb == 1 || outVisited[i])
            {
                return;
            }
            findVisibleTiles(i, solver, outVisited, outGroup);
        });
    }

    void findVisibleTiles(mswp::BoardIndex centerTileIndex, const MineSweeperSolver& solver, BoardBitMap& outVisited, TileGroup& outGroup)
    {
        outVisited[centerTileIndex] = true;
        util::applyFuncToAdjObjects<MineSweeperSolver, Tile>(centerTileIndex, solver, 
        [&](int32_t i, const Tile& tile)
        {
            if (tile.hidden() || tile.adjBombs == 0 || tile.bombProb == 1 || outVisited[i])
            {
                return;
            }
            findHiddenTiles(i, solver, outVisited, outGroup);
        });
    }

    } // namespace group end

inline uint8_t* getHardcodedCombinations(uint8_t r)
{
    static uint8_t hardCodedCombinations[] = {0, 1, 2, 3, 4, 5, 6, 7, 9, 1, 0, 2, 0, 2, 1, 3, 0, 3, 1, 3, 2, 4, 0, 4, 1, 4, 2, 4, 3, 5, 0, 5, 1, 5, 2, 5, 3, 5, 4, 6, 0, 6, 1, 6, 2, 6, 3, 6, 4, 6, 5, 7, 0, 7, 1, 7, 2, 7, 3, 7, 4, 7, 5, 7, 6, 9, 2, 1, 0, 3, 1, 0, 3, 2, 0, 3, 2, 1, 4, 1, 0, 4, 2, 0, 4, 2, 1, 4, 3, 0, 4, 3, 1, 4, 3, 2, 5, 1, 0, 5, 2, 0, 5, 2, 1, 5, 3, 0, 5, 3, 1, 5, 3, 2, 5, 4, 0, 5, 4, 1, 5, 4, 2, 5, 4, 3, 6, 1, 0, 6, 2, 0, 6, 2, 1, 6, 3, 0, 6, 3, 1, 6, 3, 2, 6, 4, 0, 6, 4, 1, 6, 4, 2, 6, 4, 3, 6, 5, 0, 6, 5, 1, 6, 5, 2, 6, 5, 3, 6, 5, 4, 7, 1, 0, 7, 2, 0, 7, 2, 1, 7, 3, 0, 7, 3, 1, 7, 3, 2, 7, 4, 0, 7, 4, 1, 7, 4, 2, 7, 4, 3, 7, 5, 0, 7, 5, 1, 7, 5, 2, 7, 5, 3, 7, 5, 4, 7, 6, 0, 7, 6, 1, 7, 6, 2, 7, 6, 3, 7, 6, 4, 7, 6, 5, 9, 3, 2, 1, 0, 4, 2, 1, 0, 4, 3, 1, 0, 4, 3, 2, 0, 4, 3, 2, 1, 5, 2, 1, 0, 5, 3, 1, 0, 5, 3, 2, 0, 5, 3, 2, 1, 5, 4, 1, 0, 5, 4, 2, 0, 5, 4, 2, 1, 5, 4, 3, 0, 5, 4, 3, 1, 5, 4, 3, 2, 6, 2, 1, 0, 6, 3, 1, 0, 6, 3, 2, 0, 6, 3, 2, 1, 6, 4, 1, 0, 6, 4, 2, 0, 6, 4, 2, 1, 6, 4, 3, 0, 6, 4, 3, 1, 6, 4, 3, 2, 6, 5, 1, 0, 6, 5, 2, 0, 6, 5, 2, 1, 6, 5, 3, 0, 6, 5, 3, 1, 6, 5, 3, 2, 6, 5, 4, 0, 6, 5, 4, 1, 6, 5, 4, 2, 6, 5, 4, 3, 7, 2, 1, 0, 7, 3, 1, 0, 7, 3, 2, 0, 7, 3, 2, 1, 7, 4, 1, 0, 7, 4, 2, 0, 7, 4, 2, 1, 7, 4, 3, 0, 7, 4, 3, 1, 7, 4, 3, 2, 7, 5, 1, 0, 7, 5, 2, 0, 7, 5, 2, 1, 7, 5, 3, 0, 7, 5, 3, 1, 7, 5, 3, 2, 7, 5, 4, 0, 7, 5, 4, 1, 7, 5, 4, 2, 7, 5, 4, 3, 7, 6, 1, 0, 7, 6, 2, 0, 7, 6, 2, 1, 7, 6, 3, 0, 7, 6, 3, 1, 7, 6, 3, 2, 7, 6, 4, 0, 7, 6, 4, 1, 7, 6, 4, 2, 7, 6, 4, 3, 7, 6, 5, 0, 7, 6, 5, 1, 7, 6, 5, 2, 7, 6, 5, 3, 7, 6, 5, 4, 9, 4, 3, 2, 1, 0, 5, 3, 2, 1, 0, 5, 4, 2, 1, 0, 5, 4, 3, 1, 0, 5, 4, 3, 2, 0, 5, 4, 3, 2, 1, 6, 3, 2, 1, 0, 6, 4, 2, 1, 0, 6, 4, 3, 1, 0, 6, 4, 3, 2, 0, 6, 4, 3, 2, 1, 6, 5, 2, 1, 0, 6, 5, 3, 1, 0, 6, 5, 3, 2, 0, 6, 5, 3, 2, 1, 6, 5, 4, 1, 0, 6, 5, 4, 2, 0, 6, 5, 4, 2, 1, 6, 5, 4, 3, 0, 6, 5, 4, 3, 1, 6, 5, 4, 3, 2, 7, 3, 2, 1, 0, 7, 4, 2, 1, 0, 7, 4, 3, 1, 0, 7, 4, 3, 2, 0, 7, 4, 3, 2, 1, 7, 5, 2, 1, 0, 7, 5, 3, 1, 0, 7, 5, 3, 2, 0, 7, 5, 3, 2, 1, 7, 5, 4, 1, 0, 7, 5, 4, 2, 0, 7, 5, 4, 2, 1, 7, 5, 4, 3, 0, 7, 5, 4, 3, 1, 7, 5, 4, 3, 2, 7, 6, 2, 1, 0, 7, 6, 3, 1, 0, 7, 6, 3, 2, 0, 7, 6, 3, 2, 1, 7, 6, 4, 1, 0, 7, 6, 4, 2, 0, 7, 6, 4, 2, 1, 7, 6, 4, 3, 0, 7, 6, 4, 3, 1, 7, 6, 4, 3, 2, 7, 6, 5, 1, 0, 7, 6, 5, 2, 0, 7, 6, 5, 2, 1, 7, 6, 5, 3, 0, 7, 6, 5, 3, 1, 7, 6, 5, 3, 2, 7, 6, 5, 4, 0, 7, 6, 5, 4, 1, 7, 6, 5, 4, 2, 7, 6, 5, 4, 3, 9, 5, 4, 3, 2, 1, 0, 6, 4, 3, 2, 1, 0, 6, 5, 3, 2, 1, 0, 6, 5, 4, 2, 1, 0, 6, 5, 4, 3, 1, 0, 6, 5, 4, 3, 2, 0, 6, 5, 4, 3, 2, 1, 7, 4, 3, 2, 1, 0, 7, 5, 3, 2, 1, 0, 7, 5, 4, 2, 1, 0, 7, 5, 4, 3, 1, 0, 7, 5, 4, 3, 2, 0, 7, 5, 4, 3, 2, 1, 7, 6, 3, 2, 1, 0, 7, 6, 4, 2, 1, 0, 7, 6, 4, 3, 1, 0, 7, 6, 4, 3, 2, 0, 7, 6, 4, 3, 2, 1, 7, 6, 5, 2, 1, 0, 7, 6, 5, 3, 1, 0, 7, 6, 5, 3, 2, 0, 7, 6, 5, 3, 2, 1, 7, 6, 5, 4, 1, 0, 7, 6, 5, 4, 2, 0, 7, 6, 5, 4, 2, 1, 7, 6, 5, 4, 3, 0, 7, 6, 5, 4, 3, 1, 7, 6, 5, 4, 3, 2, 9, 6, 5, 4, 3, 2, 1, 0, 7, 5, 4, 3, 2, 1, 0, 7, 6, 4, 3, 2, 1, 0, 7, 6, 5, 3, 2, 1, 0, 7, 6, 5, 4, 2, 1, 0, 7, 6, 5, 4, 3, 1, 0, 7, 6, 5, 4, 3, 2, 0, 7, 6, 5, 4, 3, 2, 1, 7, 6, 5, 4, 3, 2, 1, 0};
    static uint16_t rToIndex[] = {0, 0, 9, 66, 235, 516, 797, 966, 1022};
    return &hardCodedCombinations[rToIndex[r]];
}

inline uint8_t getHardcodedCombinationsSize(uint8_t n, uint8_t r)
{
    --n;
    --r;
    static uint8_t hardCodedSize[] = {1, 2, 1, 3, 3, 1, 4, 6, 4, 1, 5, 10, 10, 5, 1, 6, 15, 20, 15, 6, 1, 7, 21, 35, 35, 21, 7, 1, 8, 28, 56, 70, 56, 28, 8, 1};
    const bool valid = (n < 8 && r < 8);
    const uint8_t k = (n*(n+1)/2 + r)*valid;
    return hardCodedSize[k]*valid;
}

void sortByCombinationCount(const MineSweeperSolver& solver, group::TileGroup& outTileGroup)
{
    BoardBitMap claimed;
    for (mswp::BoardIndex i = 0; i < outTileGroup.size; i++)
    {
        // Finds max combination size
        mswp::BoardIndex maxIndex = -1;
        int8_t maxCombinationSize = -1;
        for (mswp::BoardIndex j = i; j < outTileGroup.size; j++)
        {
            group::TileWithAdjs& tile = outTileGroup.tiles[j];

            int8_t n = 0;

            util::applyFuncToAdjObjects<MineSweeperSolver, Tile>(tile.tileIndex, solver, 
            [&](int32_t adjIndex, const Tile& inTile) 
            {
                n += inTile.hidden() && !claimed[adjIndex] ;
            });
            int8_t combinationSize = getHardcodedCombinationsSize(n, solver[tile.tileIndex].adjBombs);

            if (combinationSize > maxCombinationSize)
            {
                maxCombinationSize = combinationSize;
                maxIndex = j;
            }
        }

        // Moves tile to front
        util::swap(outTileGroup.tiles[maxIndex].tileIndex, outTileGroup.tiles[i].tileIndex);

        group::TileWithAdjs& tile = outTileGroup.tiles[i];

        // Claims tiles
        util::applyFuncToAdjObjects<MineSweeperSolver, Tile>(tile.tileIndex, solver, 
        [&](int32_t adjIndex, const Tile& inTile) 
        {
            if (claimed[adjIndex] || !inTile.hidden())
            {
                return;
            }
            tile.adjTiles[tile.size] = adjIndex;
            tile.size++;
            claimed[adjIndex] = true;
        });
    }
}

int8_t getAdjBombsInSolution(const int32_t i, const int32_t width, const int32_t size, const BoardBitMap& bombLocations)
{
    int8_t adjBombs = 0;
    util::applyFuncToAdjObjects<BoardBitMap, bool>(i, width, size, bombLocations, 
    [&](const bool& isBomb) 
    {
        adjBombs += isBomb;
    });
    return adjBombs;
}

void getSolutions(group::TileGroup& outTileGroup, MineSweeperSolver& outSolver, BoardBitMap& outBombLocations, SolutionSet& outSolutionSet, const mswp::BoardIndex i, const uint8_t bombsInSolution)
{
    if (i == outTileGroup.size) // Base case
    {
        outSolutionSet.push(Solution(outBombLocations, bombsInSolution));
        return;
    }
    const group::TileWithAdjs& currTile = outTileGroup.tiles[i];
    const int8_t adjBombsInSolution = getAdjBombsInSolution(currTile.tileIndex, outSolver.width(), outSolver.size(), outBombLocations);
    const int8_t adjBombCount = outSolver[currTile.tileIndex].adjBombs - adjBombsInSolution;
    const int8_t adjClaimedCount = currTile.size;

    if (adjClaimedCount < adjBombCount || adjBombCount < 0) // Checks for invalid set
    {
        return;
    }
    if (adjClaimedCount == 0 || adjBombCount == 0) // Skips when there are 0 combinations.
    {
        getSolutions(outTileGroup, outSolver, outBombLocations, outSolutionSet, i + 1, bombsInSolution);
        return;
    }
    uint8_t* combinations = getHardcodedCombinations(adjBombCount);
    while(*combinations < adjClaimedCount)
    {
        // Used to remove solution from outBombLocations
        uint8_t* combinationsUndo = combinations;
        // Apply solution
        for (uint8_t j = 0; j < adjBombCount; j++)
        {
            const uint16_t index = currTile.adjTiles[*combinations];
            outBombLocations[index] = true;
            combinations++;
        }

        getSolutions(outTileGroup, outSolver, outBombLocations, outSolutionSet, i + 1, bombsInSolution + adjBombCount);

        // Remove solution
        for (uint8_t j = 0; j < adjBombCount; j++)
        {
            const uint16_t index = currTile.adjTiles[*combinationsUndo];
            outBombLocations[index] = false;
            combinationsUndo++;
        }
    }
}

void poppulateSolutionSets(MineSweeperSolver& outSolver, std::vector<SolutionSet>& outSolutionSet)
{
    BoardBitMap visited;
    outSolver.applyFuncToAll(
    [&](mswp::BoardIndex i, Tile& tile) 
    {
        // Disgards invalid starting tiles (0, #, @)
        if (visited[i] || tile.hidden() || tile.adjBombs == 0 || tile.bombProb == 1)
        {
            return;
        }

        group::TileGroup tileGroup;

        // Goes back and forth between find visible and 
        // find hidden tiles until a group has been fully 
        // created.
        group::findHiddenTiles(i, outSolver, visited, tileGroup);
        // Sorts combination count such that branches end 
        // earlier in NP computeProbabilities call.
        sortByCombinationCount(outSolver, tileGroup);
        // NP computeProbabilities call.
        BoardBitMap bombLocations;
        SolutionSet solutionSet;
        getSolutions(tileGroup, outSolver, bombLocations, solutionSet);
        outSolutionSet.emplace_back(std::move(solutionSet));
    });
}

uint64_t getTotalAmountOfSolutions(const SolutionSet& solutionSet)
{
    uint64_t totalAmountOfSolutions = 0;
    for (const Solution& solution : solutionSet)
    {
        totalAmountOfSolutions += solution.numberOfSolutions;
    }
    return totalAmountOfSolutions;
}

void calculateProbs(MineSweeperSolver& outSolver, TileProbs& outProbs)
{
    std::vector<SolutionSet> solutionSets;
    poppulateSolutionSets(outSolver, solutionSets);

    int32_t diff = outSolver.remainingBombs() - outSolver.remainingDeepTiles();
    mswp::FlagsRemaining minBombs = std::max(diff, 0);
    mswp::FlagsRemaining maxBombs = outSolver.remainingBombs();
    bool combined = shouldCombineSolutionSets(solutionSets, minBombs, maxBombs);
    if (combined)
    {
        combineSolutionSets(solutionSets, minBombs, maxBombs);
    }

    mswp::BoardSize size = outSolver.size();
    std::fill(outProbs.begin(), outProbs.end(), INFINITY);


    // TODO: Cache indicies for optimization.

    double remainingProb = outSolver.remainingBombs();

    for (const SolutionSet& solutionSet : solutionSets)
    {
        uint64_t totalAmountOfSolutions = solutionSet.size();
        if (combined)
        {
            totalAmountOfSolutions = getTotalAmountOfSolutions(solutionSet);
        }
        for (const Solution& solution : solutionSet)
        {
            for (mswp::BoardIndex i = 0; i < size; i++)
            {
                if (!outSolver.isNotDeepTile()[i])
                {
                    continue;
                }
                // Zero if not a bomb, solution.numberOfSolutions if bomb.
                uint64_t solutionCount = (solution.numberOfSolutions + !combined) * solution.solution[i];

                double prob = static_cast<double>(solutionCount) / totalAmountOfSolutions;

                if (outProbs[i] == INFINITY && prob != 0)
                {
                    outProbs[i] = 0;
                }

                // Gets probability of tile being a bomb
                outProbs[i] += prob;
                remainingProb -= prob;
            }
        }
    }

    double deepTileProb = remainingProb / outSolver.remainingDeepTiles();

    for (mswp::BoardIndex i = 0; i < size; i++)
    {
        if (!outSolver.isNotDeepTile()[i])
        {
            outProbs[i] = deepTileProb;
        }
    }
}

} // namespace slvr end