from buffer import ReplayBuffer
from model import Model, soft_update, hard_update
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import datetime
import time
from torch.utils.tensorboard import SummaryWriter
import random
import os
import pygame
import sys

class Agent():

    def __init__(self, env, hidden_layer, learning_rate, step_repeat, gamma) -> None:

        self.env = env

        self.step_repeat = step_repeat

        self.gamma = gamma

        obs, info = self.env.reset()
        
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.memory = ReplayBuffer(max_size=500000, input_shape=[13, 8, 8], n_actions=env.action_space.n, device=self.device)

        self.model = Model(action_dim=env.action_space.n, hidden_dim=hidden_layer).to(self.device)

        self.target_model = Model(action_dim=env.action_space.n, hidden_dim=hidden_layer).to(self.device)

        # Initialize target networks with model parameters
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer_1 = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.learning_rate = learning_rate

        print(f"Initialized agents on device: {self.device}")


    def train(self, episodes, max_episode_steps, summary_writer_suffix, batch_size, epsilon, epsilon_decay, min_epsilon):
        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{summary_writer_suffix}'
        writer = SummaryWriter(summary_writer_name)

        if not os.path.exists('models'):
            os.makedirs('models')

        total_steps = 0

        for episode in range(episodes):

            done = False
            episode_reward = 0
            obs, info = self.env.reset()

            episode_steps = 0

            episode_start_time = time.time()

            while not done and episode_steps < max_episode_steps:

                player = self.env.get_current_player()
                
                if random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    if(player == 0): # Take actions for player 0. 
                        q_values = self.model.forward(obs.to(self.device), 
                                                      torch.tensor([player]).to(self.device).unsqueeze(1))
                        action = torch.argmax(q_values, dim=-1).item()
                    else:
                        action = self.env.action_space.sample()

                    action = torch.argmax(q_values, dim=-1).item()

                reward = 0

                for i in range(self.step_repeat):
                    reward_temp = 0
                    next_obs, reward_temp, done, info = self.env.step(action=action)

                    reward += reward_temp

                    if(done):
                        break

                
                if(player == 1): # If player is 1, mark as "enemy" and invert rewards.
                    reward *= -1

                self.memory.store_transition(obs, action, reward, next_obs, done, player)

                obs = next_obs        

                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                if self.memory.can_sample(batch_size):
                    observations, actions, rewards, next_observations, dones = self.memory.sample_buffer(batch_size)

                    dones = dones.unsqueeze(1).float()

                    # Current Q-values from both models
                    print(observations)
                    q_values = self.model(observations)
                    actions = actions.unsqueeze(1).long()
                    qsa_batch = q_values.gather(1, actions)

                    # Action selection using the main models
                    next_actions = torch.argmax(self.model(next_observations), dim=1, keepdim=True)

                    # Q-value evaluation using the target models
                    next_q_values = self.target_model(next_observations).gather(1, next_actions)

                    # Compute the target using Double DQN with minimization
                    target_b = rewards.unsqueeze(1) + (1 - dones) * self.gamma * next_q_values

                    # Calculate the loss for both models
                    loss = F.mse_loss(qsa_batch, target_b.detach())

                    writer.add_scalar("Loss/model", loss.item(), total_steps)

                    # Backpropagation and optimization step for both models
                    self.model.zero_grad()
                    loss.backward()
                    self.optimizer_1.step()

                    # Update the target models periodically
                    if episode_steps % 4 == 0:
                        soft_update(self.target_model, self.model)

            self.model.save_the_model()

            writer.add_scalar('Score', episode_reward, episode)
            writer.add_scalar('Epsilon', epsilon, episode)

            if epsilon > min_epsilon:
                epsilon *= epsilon_decay

            episode_time = time.time() - episode_start_time

            print(f"Completed episode {episode} with score {episode_reward}")
            print(f"Episode Time: {episode_time:1f} seconds")
            print(f"Episode Steps: {episode_steps}")
