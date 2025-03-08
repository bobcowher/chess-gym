import chess
import numpy as np
import gym
from gym import spaces

class ChessEnv(gym.Env):
    def __init__(self, render_mode='rgb_array'):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.action_space = spaces.Discrete(4672)  # Max possible moves in a game
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8, 12), dtype=np.float32)
        
        assert(render_mode == 'rgb_array' or render_mode == 'human', "Render mode should be rgb_aray or human")


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

    def render(self, mode='human'):
        print(self.board)

