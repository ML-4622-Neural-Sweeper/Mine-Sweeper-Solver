import MineSweeper as ms
import numpy as np

# Constants
WIDTH = 30
HEIGHT = 16
BOMB_COUNT = 99

class MineSweeperEnv:
    """
        Enviroment containing Minesweeper Board and solver, requires action input when it 
        comes up against non deterministic board configuration.
    """
    def __init__(
        self, reward_per_tile_clicked: float, reward_for_winning: float, reward_for_losing: float
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
        """
        global WIDTH, HEIGHT, BOMB_COUNT

        # Init Board
        self.board = ms.MineSweeper(WIDTH, HEIGHT, BOMB_COUNT)
        self.solver = ms.MineSweeperSolver(self.board)

        # Init rewards
        self.reward_per_tile_clicked = reward_per_tile_clicked
        self.reward_for_winning = reward_for_winning
        self.reward_for_losing = reward_for_losing

    def take_action(self, index: int) -> tuple[float, bool]:
        """
            Preforms action on enviroment.
            Args:
                index:
                    Index to click, chosen by RL Model.
            Returns:
                Tuple containing reward of action, and whether or not the game is over.
        """
        reward = ms.get_reward(
            index, self.reward_per_tile_clicked, self.reward_for_winning, 
            self.reward_for_losing, self.board, self.solver
        )
        is_game_over = ms.board.game_state() == ms.GameState.WON or ms.board.game_state() == ms.GameState.LOST
        return (reward, is_game_over) 

    def get_board_data(self) -> np.ndarray:
        """
            Gets data used by RL model.
            Returns:
                Float Matrix containing RL data.
        """
        global WIDTH, HEIGHT, BOMB_COUNT

        IS_BOMB_INDEX    = 0
        ADJ_BOMBS_INDEX  = 1
        ADJ_HIDDEN_INDEX = 2
        STATE_INDEX      = 3

        # 4 x WIDTH*HEIGHT matrix containing info about baord.
        board_state = np.zeros((4,WIDTH*HEIGHT), dtype=float)

        for tile in self.solver.tiles():
            # 1.0 for bomb, 0.0 for not a bomb
            board_state[IS_BOMB_INDEX]    = float(tile.is_bomb())
            # [0, 8] in N for visible tiles, -1 for hidden tiles.
            board_state[ADJ_BOMBS_INDEX]  = float(tile.adj_bombs)
            # [0, 8] in N for hidden tiles, -1 for hidden tiles.
            board_state[ADJ_HIDDEN_INDEX] = float(tile.adj_hidden)
        
        # [0, BOMB_COUNT] in N, amount of bombs left.
        board_state[STATE_INDEX][0] = float(self.board.flags_remaining())
        # [0, MAX_TILE_COUNT] in N, amount of tiles left that do not board a visible one.
        board_state[STATE_INDEX][1] = float(self.solver.deep_tiles_remaining())
        # WIDTH Constant
        board_state[STATE_INDEX][2] = float(WIDTH)
        # HEIGHT Constant
        board_state[STATE_INDEX][3] = float(HEIGHT)