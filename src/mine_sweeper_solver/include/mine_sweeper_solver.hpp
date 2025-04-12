#pragma once

#include <cstdint>
#include <cmath>
#include <array>
#include <functional>
#include <bitset>

#include "mine_sweeper_constants.hpp"
#include "mine_sweeper.hpp"
#include "mine_sweeper_solver_constants.hpp"

namespace slvr
{

struct Tile
{
    constexpr Tile() : bombProb{NAN}, adjBombs{-1}, adjUnknowns{-1} {}
    constexpr Tile(float bombProb, int8_t adjBombs, int8_t adjUnknowns) : bombProb{bombProb}, adjBombs{adjBombs}, adjUnknowns{adjUnknowns} {}
    constexpr bool operator==(const Tile& other) const
    {
        return adjBombs == other.adjBombs && adjUnknowns == other.adjUnknowns && bombProb == other.bombProb;
    }
    constexpr bool operator!=(const Tile& other) const
    {
        return adjBombs != other.adjBombs || adjUnknowns != other.adjUnknowns || bombProb != other.bombProb;
    }
    bool hidden() const
    {
        return bombProb != bombProb;
    }
    bool isBomb() const
    {
        return bombProb == 1;
    }
    float bombProb;
    int8_t adjBombs;
    int8_t adjUnknowns;
};

typedef std::array<Tile, MSWP_MAX_TILES> Tiles;
typedef std::array<mswp::BoardIndex, MSWP_MAX_TILES> TilesWithAdjBombBuffer;
typedef mswp::BoardIndex TilesWithAdjBombSize;
typedef std::bitset<MSWP_MAX_TILES> BoardBitMap;
typedef mswp::BoardIndex BoardBitSize;
typedef std::initializer_list<Tile> SolverInitList;

class MineSweeperSolver
{
public:
    MineSweeperSolver(const mswp::MineSweeper& mineSweeper);
    void updateTile(const mswp::TileStringIndex i, bool unflag = false);
    void update(const mswp::TileString& otherTileString);

    void applyFuncToAll(std::function<void(const mswp::BoardIndex i, Tile& outTile)> func);
    void applyFuncToAll(std::function<void(Tile& outTile)> func);
    void applyFuncToAll(std::function<void(const mswp::BoardIndex i)> func);

    void applyFuncToTilesWithAdjBombs(std::function<void(const TilesWithAdjBombSize i, Tile& outTile)> func);
    void applyFuncToTilesWithAdjBombs(std::function<void(Tile& outTile)> func);
    void applyFuncToTilesWithAdjBombs(std::function<void(const TilesWithAdjBombSize i)> func);

    inline Tile& operator[](mswp::BoardIndex i)
    {
        return m_Tiles[i];
    }
    inline const Tile& operator[](mswp::BoardIndex i) const
    {
        return m_Tiles[i];
    }
    
    inline mswp::BoardWidth width() const
    {
        return m_Width;
    }
    inline mswp::BoardSize size() const
    {
        return m_Size;
    }
    
    inline const mswp::TileString& tileString() const
    {
        return m_TileString;
    }
    inline mswp::FlagsRemaining remainingBombs() const
    {
        return m_RemainingBombs;
    }
    inline mswp::BoardSize remainingDeepTiles() const
    {
        return m_RemainingDeepTiles;
    }
    
    inline const Tiles& tiles() const
    {
        return m_Tiles;
    }
    
    inline const TilesWithAdjBombBuffer& tilesWithAdjBombBuffer() const
    {
        return m_TilesWithAdjBombBuffer;
    }
    inline TilesWithAdjBombSize tilesWithAdjBombBufferSize() const
    {
        return m_TilesWithAdjBombBufferSize;
    }
    
    inline mswp::TileString& tileString()
    {
        return m_TileString;
    }
    inline mswp::FlagsRemaining& remainingBombs()
    {
        return m_RemainingBombs;
    }
    inline mswp::BoardSize& remainingDeepTiles()
    {
        return m_RemainingDeepTiles;
    }
    
    inline Tiles& tiles()
    {
        return m_Tiles;
    }
    
    inline TilesWithAdjBombBuffer& tilesWithAdjBombBuffer()
    {
        return m_TilesWithAdjBombBuffer;
    }
    inline TilesWithAdjBombSize& tilesWithAdjBombBufferSize()
    {
        return m_TilesWithAdjBombBufferSize;
    }
    inline const BoardBitMap& isNotDeepTile() const
    {
        return m_IsNotDeepTile;
    }

    bool operator==(const MineSweeperSolver& other) const;
    bool operator==(const SolverInitList& other) const;

private:
    const mswp::BoardWidth m_Width;
    const mswp::BoardSize m_Size;
    
    mswp::TileString m_TileString;
    mswp::FlagsRemaining m_RemainingBombs;
    mswp::BoardSize m_RemainingDeepTiles;
    Tiles m_Tiles;

    TilesWithAdjBombBuffer m_TilesWithAdjBombBuffer;
    TilesWithAdjBombSize m_TilesWithAdjBombBufferSize;
    
    BoardBitMap m_IsNotDeepTile;
};

std::ostream& operator<<(std::ostream &out, const Tile& tile);
std::ostream& operator<<(std::ostream &out, const MineSweeperSolver& mineSweeperSolver);

} // namespace slvr end