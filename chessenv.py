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

class ChessEnv(gym.Env):
    def __init__(self, render_mode='rgb_array'):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.action_space = spaces.Discrete(4672)  # Max possible moves in a game
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8, 12), dtype=np.float32)
        
        if (render_mode != 'none') and (render_mode != 'human') and (render_mode != 'ascii'):
            raise ValueError(f"Render mode must be either rgb_array or human. Render mode was passed in as {render_mode}")
        
        self.render_mode = render_mode

        if(self.render_mode == 'human'):
            self.width = 600
            self.height = 600
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Chess RL Environment")
            self.running = True

    def reset(self):
        self.board.reset()
        return self._get_obs()
    
    def step(self, action):
        move = self._decode_action(action)
        if move in self.board.legal_moves:
            self.board.push(move)
            reward = self._get_reward()
            done = self.board.is_game_over()
            return self._get_obs(), reward, done, {}
        else:
            return self._get_obs(), -1, True, {}  # Illegal move penalty
    
    def _decode_action(self, action):
        moves = list(self.board.legal_moves)
        if action < len(moves):
            return moves[action]
        return moves[0]  # Default to first move (should not happen in a well-trained model)
    
    def _get_obs(self):
        return self._board_to_array()
    
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
            return 1 if self.board.turn == chess.BLACK else -1
        return 0


    def render(self):
        if self.render_mode == "human":
            svg = chess.svg.board(self.board, size=350)  # Generates an SVG
            png = BytesIO()
            cairosvg.svg2png(bytestring=svg.encode('utf-8'), write_to=png)  # Convert SVG to PNG
            image = Image.open(png)
            plt.imshow(image)
            plt.axis("off")
            plt.show()
        elif self.render_mode == "ascii":
            print(self.board)

        # Note. Other render modes should be used for training.
