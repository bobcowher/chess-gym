import chess
import chess.svg
import numpy as np
import gym
from gym import spaces
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import cairosvg
import pygame
import torch
import random

class ChessEnv(gym.Env):
    def __init__(self, render_mode='rgb_array'):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.action_space = spaces.Discrete(4672)  # Max possible moves in a game
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8, 12), dtype=np.float32)
        
        if (render_mode != 'none') and (render_mode != 'human') and (render_mode != 'ascii'):
            raise ValueError(f"Render mode must be either rgb_array or human. Render mode was passed in as {render_mode}")
        
        self.render_mode = render_mode
        self.white = chess.WHITE
        self.black = chess.BLACK

        if(self.render_mode == 'human'):
            pygame.init()
            self.width = 600
            self.height = 600
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Chess RL Environment")
            self.running = True

    def get_current_player(self):
        if(chess.WHITE == self.board.turn):
            return 0
        else:
            return 1
        # white is represented by a 0, black by 1

    def reset(self):
        self.board.reset()
        return self._get_obs(), self._get_info()
    

    def step(self, action):
        """ Executes a move in the environment. Returns next state, reward, done flag, and info. """
        
        reward = 0

        legal_moves = list(self.board.legal_moves)

        # Decode action into a valid chess move
        if action < len(legal_moves):
            move = legal_moves[action]
        else:
            move = random.choice(legal_moves) if legal_moves else None  # Avoid crash if no moves exist
            reward += -1

        if move and move in legal_moves:
            self.board.push(move)  # Apply move
            reward += self._get_reward()
            done = self.board.is_game_over()

            # Debugging: Identify why the game ended
            if done:
                termination_reason = "Checkmate" if self.board.is_checkmate() else \
                                    "Stalemate" if self.board.is_stalemate() else \
                                    "Threefold Repetition" if self.board.is_repetition(3) else \
                                    "50-Move Rule" if self.board.is_fifty_moves() else "Unknown"
                print(f"Game Over! Reason: {termination_reason}")

            return self._get_obs(), reward, done, self._get_info()

        else:
            print("Illegal move attempted! Penalizing agent and terminating episode.")
            return self._get_obs(), -1, True, self._get_info()  # Penalize illegal move


    def _get_obs(self):
        board_tensor = torch.tensor(self._board_to_array())
        player_channel = np.full((8, 8, 1), self.board.turn == chess.WHITE, dtype=np.float32)
        # board_tensor = torch.tensor(obs).permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 12, 8, 8)
        obs_with_player = torch.tensor(np.concatenate([board_tensor, player_channel], axis=-1))
        obs = obs_with_player.permute(2, 0, 1).unsqueeze(0)

        return obs 

    def _get_info(self):
        return {'current_player': self.get_current_player()}

    def _board_to_array(self):
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3,
            chess.QUEEN: 4, chess.KING: 5
        }
        board_array = np.zeros((8, 8, 12), dtype=np.float32)
        
        for square, piece in self.board.piece_map().items():
            x, y = divmod(square, 8)
            idx = piece_map[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
            board_array[x, y, idx] = 1.0

        return board_array
    
    def _get_reward(self):
        if self.board.is_checkmate():
            return 1
        elif self.board.is_stalemate():
            return -0.5  # Penalize getting stuck

        elif self.board.is_repetition(3):
            return -0.5  # Penalize repeating positions

        elif self.board.is_fifty_moves():
            return -0.5  # Penalize getting stuck in a 50-move rule draw
        else:
            return 0

    def render(self):
        if self.render_mode == "human":
            # Process Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Convert chess SVG to PNG using cairosvg
            svg = chess.svg.board(self.board, size=600)
            png_bytes = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
            
            # Load image using PIL
            image = Image.open(BytesIO(png_bytes))
            mode = image.mode
            size = image.size
            data = image.tobytes()

            # Convert to Pygame surface
            pygame_image = pygame.image.fromstring(data, size, mode)
            self.screen.blit(pygame_image, (0, 0))
            pygame.display.flip()
        elif self.render_mode == "ascii":
            print(self.board)

        # Note. Other render modes should be used for training.
