from agent import Agent
from chessenv import ChessEnv

episodes = 10000
max_episode_steps = 10000
total_steps = 0
step_repeat = 4
max_episode_steps = max_episode_steps / step_repeat

batch_size = 64
learning_rate = 0.0001
epsilon = 1.0
min_epsilon = 0.1
epsilon_decay = 0.995
gamma = 0.99

hidden_layer = 512

# print(observation.shape)

# Constants

env = ChessEnv(render_mode='human')
env.reset()


summary_writer_suffix = f'dqn_lr={learning_rate}_hl={hidden_layer}_mse_loss_bs={batch_size}_double_dqn'

agent = Agent(env, hidden_layer=hidden_layer,
              learning_rate=learning_rate, step_repeat=step_repeat,
              gamma=gamma)


agent.test()