import numpy as np
import csv
import random
import torch

class CPUReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions, device='cpu'):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.uint8)
        self.next_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.uint8)
        self.action_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        self.device = device


    def can_sample(self, batch_size):
        if self.mem_ctr > (batch_size * 5):
            return True
        else:
            return False

    def store_transition(self, state, action, reward, next_state, done, player):
        index = self.mem_ctr % self.mem_size

        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = torch.as_tensor(action, dtype=torch.long, device=self.device).view(1)
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_ctr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        next_states = self.next_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        # Convert to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        return states, actions, rewards, next_states, dones 


class ReplayBufferGPU:
    def __init__(self, max_size, input_shape, n_actions, device='cpu'):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.device = torch.device(device)

        self.state_memory = torch.zeros((self.mem_size, *input_shape), dtype=torch.uint8, device=self.device)
        self.next_state_memory = torch.zeros((self.mem_size, *input_shape), dtype=torch.uint8, device=self.device)
        self.action_memory = torch.zeros((self.mem_size, 1), dtype=torch.long, device=self.device)
        self.reward_memory = torch.zeros(self.mem_size, dtype=torch.float32, device=self.device)
        self.terminal_memory = torch.zeros(self.mem_size, dtype=torch.bool, device=self.device)

    def can_sample(self, batch_size):
        return self.mem_ctr > (batch_size * 5)

    def store_transition(self, state, action, reward, next_state, done, player=None):
        index = self.mem_ctr % self.mem_size

        self.state_memory[index] = torch.as_tensor(state, dtype=torch.uint8, device=self.device)
        self.next_state_memory[index] = torch.as_tensor(next_state, dtype=torch.uint8, device=self.device)

        # Store discrete action as [1] long tensor
        self.action_memory[index] = torch.as_tensor(action, dtype=torch.long, device=self.device).flatten()[:1]

        self.reward_memory[index] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.terminal_memory[index] = torch.tensor(done, dtype=torch.bool, device=self.device)

        self.mem_ctr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)
        batch = torch.randint(0, max_mem, (batch_size,), device=self.device)

        states = self.state_memory[batch].float()  # normalize later if needed
        next_states = self.next_state_memory[batch].float()
        actions = self.action_memory[batch].view(-1, 1)
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch].float()

        return states, actions, rewards, next_states, dones
