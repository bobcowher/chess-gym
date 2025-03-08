import gym
from chessenv import ChessEnv
import time
from model import Model
import torch

# Example usage
#env = ChessEnv(render_mode='rgb_array')
env = ChessEnv(render_mode='human')
obs = env.reset()
env.render()

done = False

print("Action Space: ", env.action_space.n)

model = Model(action_dim=env.action_space.n, hidden_dim=256)


while not done:

    player = env.get_current_player()
    print(env.board.turn)
    if(player == 1):
        q_values = model.forward(obs, player)
        action = torch.argmax(q_values, dim=-1).item()
    else:
        action = env.action_space.sample()

    obs, reward, done, info = env.step(action)

    env.render()
    print(obs)
    print('-' * 20)
    time.sleep(1)