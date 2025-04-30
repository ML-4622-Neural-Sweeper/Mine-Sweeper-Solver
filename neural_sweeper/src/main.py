from mine_sweeper_env import MineSweeperEnv
from hyperparameters import *
from agent import DQNAgent

def main():
    print('Initializing env...')
    # Create agent and env
    env = MineSweeperEnv(
        REWARD_PER_TILE_CLICKED, REWARD_FOR_WINNING, REWARD_FOR_LOSING, 
        REWARD_FOR_CLICKING_VISIBLE_TILE, REWARD_FOR_CLICKING_FLAGGED_TILE
    )
    print('Done!')
    print('Initializing agent...')
    agent = DQNAgent()
    print('Done!')
    print(f'Running {EPISODE_COUNT} amount of episodes on model...')
    # Run model
    for episode_count in range(EPISODE_COUNT):
        print(f'Starting episode {episode_count}...')
        # Init episode reward
        episode_reward = 0
        # Init state
        state = env.get_state()
        # Trains model
        done = False
        while not done:
            # Get action
            action = agent.act(state)
            # Preform action, get reward
            reward, done = env.step(action)

            # Get next state
            next_state = env.get_state()

            # Add step to replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            
            # Train model
            agent.train_step()

            # Tracks total reward of episode
            episode_reward += reward

        env.reset()

        # Syncs deep q models (expensive!)
        if episode_count % TARGET_UPDATE_FREQ == 0: 
            agent.update_target()

        # Reduces exploration
        agent.decay_epsilon()

        # Logs progress
        print(f'Done! Episode {episode_count}: Reward = {episode_reward}, Epsilon = {agent.epsilon}')

if __name__ == '__main__':
    main()