from mine_sweeper_env import MineSweeperEnv, ms
from hyperparameters import *
from agent import DQNAgent
import torch

def main():
    # Create agent and env
    env = MineSweeperEnv(
        REWARD_PER_TILE_CLICKED, REWARD_FOR_WINNING, REWARD_FOR_LOSING, 
        REWARD_FOR_CLICKING_VISIBLE_TILE, REWARD_FOR_CLICKING_FLAGGED_TILE, seed=0
    )
    agent = DQNAgent()
    agent.model.load_state_dict(torch.load('model_weights.pth'))
    agent.epsilon = 0.0
    # Run model
    won, lost = 0, 0
    for i in range(1000):
        # Init state
        state = env.get_state()
        done = False
        while not done:
            # Get mask
            action_mask = env.get_action_mask()
            # Get action
            action = agent.act(state, action_mask)
            # Preform action, get reward
            _, done = env.step(action)
            # Get next state
            state = env.get_state()
        if env.board.game_state() == ms.GameState.WON:
            won += 1
        else:
            lost += 1
        env.reset(seed=i+1)
    print(f'AI DONE: Games Won: {won}, Games Lost: {lost}, Percentage Won: {float(won) / (won + lost)}')
    # Run game
    won, lost = 0, 0
    flag_actions = ms.ActionArray()
    click_actions = ms.ActionArray()
    env.reset(seed=0)
    for i in range(1000):
        print(f'Game {i}')
        done = False
        click_actions.push(0)
        while env.board.game_state() != ms.GameState.WON and env.board.game_state() != ms.GameState.LOST:
            ms.recommended_actions(env.solver, click_actions, flag_actions)
            ms.use_action_arrays(click_actions, flag_actions, env.board)
            env.solver.update(env.board.tile_string())

        if env.board.game_state() == ms.GameState.WON:
            won += 1
        else:
            lost += 1
        env.reset(seed=i+1)
    print(f'DONE: Games Won: {won}, Games Lost: {lost}, Percentage Won: {float(won) / (won + lost)}')

if __name__ == '__main__':
    main()