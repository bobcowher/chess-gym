import gym
from chessenv import ChessEnv
import time


# Example usage
env = ChessEnv(render_mode='frank')
obs = env.reset()
env.render()

done = False

while not done:

    action = env.action_space.sample()

    obs, reward, done, info = env.step(action)

    env.render()
    print('-' * 20)
    time.sleep(1)