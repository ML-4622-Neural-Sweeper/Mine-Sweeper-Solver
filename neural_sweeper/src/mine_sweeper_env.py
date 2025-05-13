import MineSweeper as ms
from hyperparameters import *

import torch

class MineSweeperEnv:
    """
        Enviroment containing Minesweeper Board and solver, requires action input when it 
        comes up against non deterministic board configuration.
    """
    def __init__(
        self, reward_per_tile_clicked: float, reward_for_winning: float, reward_for_losing: float, 
        reward_for_clicking_visible_tile: float, reward_for_clicking_flagged_tile: float, seed=None
    ):
        """
            Creates MineSweeperEnv object used for reinforcement learning.
            Args:
                reward_per_tile_clicked:
                    Amount of reward per tile clicked.
                reward_for_winning:
                    Amount of reward when game won.
                reward_for_losing:
                    Amount of reward when game lost.
                reward_for_clicking_visible_tile:
                    Amount of reward gained when visible tile is clicked.
                reward_for_clicking_flagged_tile:
                    Amount of reward gained when flagged tile is clicked.
        """
        global BOARD_WIDTH, BOARD_HEIGHT, BOMB_COUNT

        # Init Board
        if seed is None:
            self.board = ms.MineSweeper(BOARD_WIDTH, BOARD_HEIGHT, BOMB_COUNT)
        else:
            self.board = ms.MineSweeper(BOARD_WIDTH, BOARD_HEIGHT, BOMB_COUNT, seed)
        self.solver = ms.MineSweeperSolver(self.board)

        # Init rewards
        self.reward_per_tile_clicked = reward_per_tile_clicked
        self.reward_for_winning = reward_for_winning
        self.reward_for_losing = reward_for_losing
        self.reward_for_clicking_visible_tile = reward_for_clicking_visible_tile
        self.reward_for_clicking_flagged_tile = reward_for_clicking_flagged_tile

    def get_action_mask(self) -> torch.BoolTensor:
        """
        Returns a mask of legal actions. True means the action is valid.
        """
        tiles = self.solver.tiles()
        mask = torch.ones(BOARD_WIDTH * BOARD_HEIGHT, dtype=torch.bool)
        for i in range(BOARD_WIDTH * BOARD_HEIGHT):
            mask[i] = tiles[i].hidden() and not tiles[i].is_bomb()
        return mask
    
    def step(self, index: int) -> tuple[float, bool]:
        """
            Preforms action on enviroment.
            Args:
                index:
                    Index to click, chosen by RL Model.
            Returns:
                Tuple containing reward of action, and whether or not the game is over.
        """
        print('get reward!')
        reward = ms.get_reward(
            index, self.reward_per_tile_clicked, self.reward_for_winning, 
            self.reward_for_losing, self.reward_for_clicking_visible_tile,
            self.reward_for_clicking_flagged_tile, self.board, self.solver
        )
        print('got reward!')
        is_game_over = self.board.game_state() == ms.GameState.WON or self.board.game_state() == ms.GameState.LOST
        return (reward, is_game_over)
    
    def reset(self, seed=None):
        global BOARD_WIDTH, BOARD_HEIGHT, BOMB_COUNT

        # Reset Boards
        if seed is None:
            self.board = ms.MineSweeper(BOARD_WIDTH, BOARD_HEIGHT, BOMB_COUNT)
        else:
            self.board = ms.MineSweeper(BOARD_WIDTH, BOARD_HEIGHT, BOMB_COUNT, seed)
        self.solver = ms.MineSweeperSolver(self.board)

    def get_state(self) -> torch.FloatTensor:
        """
            Gets data used by RL model.
            Returns:
                Float Matrix containing RL data.
        """
        global BOARD_WIDTH, BOARD_HEIGHT, BOMB_COUNT

        IS_BOMB_INDEX    = 0
        ADJ_BOMBS_INDEX  = 1
        ADJ_HIDDEN_INDEX = 2
        STATE_INDEX      = 3

        # 4 x WIDTH*HEIGHT matrix containing info about baord.
        board_state = torch.zeros((4,BOARD_WIDTH*BOARD_HEIGHT), dtype=torch.float32)
        tiles = self.solver.tiles()

        for i in range(BOARD_WIDTH*BOARD_HEIGHT):
            # 1.0 for bomb, 0.0 for not a bomb
            board_state[IS_BOMB_INDEX][i]    = tiles[i].is_bomb()
            # [0, 8] in N for visible tiles, -1 for hidden tiles.
            board_state[ADJ_BOMBS_INDEX][i]  = tiles[i].adj_bombs
            # [0, 8] in N for hidden tiles, -1 for hidden tiles.
            board_state[ADJ_HIDDEN_INDEX][i] = tiles[i].adj_hidden
        
        # [0, BOMB_COUNT] in N, amount of bombs left.
        board_state[STATE_INDEX][0] = float(self.board.flags_remaining())
        # [0, MAX_TILE_COUNT] in N, amount of tiles left that do not board a visible one.
        board_state[STATE_INDEX][1] = float(int(self.solver.deep_tiles_remaining()))
        # WIDTH Constant
        board_state[STATE_INDEX][2] = float(BOARD_WIDTH)
        # HEIGHT Constant
        board_state[STATE_INDEX][3] = float(BOARD_HEIGHT)

        return board_state