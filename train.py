from agent import Agent
from chessenv import ChessEnv

episodes = 50000
max_episode_steps = 10000
total_steps = 0
batch_size = 64
learning_rate = 0.0001
epsilon = 1.0
min_epsilon = 0.1
epsilon_decay = 0.995
gamma = 0.99
tau = 0.005

hidden_layer = 1024

# print(observation.shape)

# Constants

env = ChessEnv(render_mode='none')


summary_writer_suffix = f'dqn_lr={learning_rate}_hl={hidden_layer}_bs={batch_size}_t={tau}'

agent = Agent(env, hidden_layer=hidden_layer,
              learning_rate=learning_rate,
              gamma=gamma)


# Training Phase 1

agent.train(episodes=episodes, max_episode_steps=max_episode_steps, summary_writer_suffix=summary_writer_suffix + "-phase-1",
            batch_size=batch_size, epsilon=epsilon, epsilon_decay=epsilon_decay,
            min_epsilon=min_epsilon, tau=tau)
    

