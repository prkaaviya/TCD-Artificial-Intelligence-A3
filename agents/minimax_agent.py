import numpy as np
import time
from typing import Tuple, List, Dict, Any, Union, Optional

class MinimaxAgent:
    """
    Minimax agent with optional alpha-beta pruning.
    
    This agent uses the minimax algorithm to select the best action.
    It can be configured to use alpha-beta pruning for efficiency.
    For Connect4, a depth limit and heuristic evaluation function are used.
    """
    
    def __init__(self, player_id: int, use_alpha_beta: bool = True, max_depth: Optional[int] = None):
        """
        Initialize the Minimax agent.
        
        Args:
            player_id: The ID of the player (1 or 2)
            use_alpha_beta: Whether to use alpha-beta pruning
            max_depth: Maximum depth to search (None for unlimited)
        """
        self.player_id = player_id
        self.opponent_id = 3 - player_id
        self.use_alpha_beta = use_alpha_beta
        self.max_depth = max_depth
        self.name = f"Minimax {'with' if use_alpha_beta else 'without'} Alpha-Beta"
        if max_depth:
            self.name += f" (Depth {max_depth})"
        
        # For performance tracking
        self.nodes_visited = 0
        self.execution_time = 0
    
    def select_action(self, game, training: bool = False) -> Union[Tuple[int, int], int]:
        """
        Select the best action using minimax.
        
        Args:
            game: The game object
            
        Returns:
            The best action
        """
        # Reset performance tracking
        self.nodes_visited = 0
        start_time = time.time()
        
        valid_actions = game.get_valid_actions()
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        # In case we run out of time/depth, have a fallback move
        best_action = valid_actions[0]
        
        if self.use_alpha_beta:
            best_value = float('-inf')
            alpha = float('-inf')
            beta = float('inf')
            
            for action in valid_actions:
                game_copy = self._copy_game(game)
                _, _, done, info = game_copy.step(action)
                
                # If this move wins immediately, take it
                if done and info["winner"] == self.player_id:
                    self.execution_time = time.time() - start_time
                    return action
                
                # Otherwise evaluate the move
                if done:
                    value = 0  # Draw
                else:
                    value = self._min_value(game_copy, 1, alpha, beta)
                
                if value > best_value:
                    best_value = value
                    best_action = action
                
                alpha = max(alpha, best_value)
        else:
            best_value = float('-inf')
            
            for action in valid_actions:
                game_copy = self._copy_game(game)
                _, _, done, info = game_copy.step(action)
                
                # If this move wins immediately, take it
                if done and info["winner"] == self.player_id:
                    self.execution_time = time.time() - start_time
                    return action
                
                # Otherwise evaluate the move
                if done:
                    value = 0  # Draw
                else:
                    value = self._min_value_no_pruning(game_copy, 1)
                
                if value > best_value:
                    best_value = value
                    best_action = action
        
        self.execution_time = time.time() - start_time
        return best_action
    
    def _max_value(self, game, depth: int, alpha: float, beta: float) -> float:
        """Maximizing player in minimax with alpha-beta pruning."""
        self.nodes_visited += 1
        
        # Check if we've reached the maximum depth or a terminal state
        if (self.max_depth is not None and depth >= self.max_depth) or game.is_terminal():
            return self._evaluate(game)
        
        value = float('-inf')
        valid_actions = game.get_valid_actions()
        
        for action in valid_actions:
            game_copy = self._copy_game(game)
            _, _, done, info = game_copy.step(action)
            
            if done:
                if info["winner"] == self.player_id:
                    child_value = 1.0  # Win
                elif info["winner"] == 0:
                    child_value = 0  # Draw
                else:
                    child_value = -1.0  # Loss
            else:
                child_value = self._min_value(game_copy, depth + 1, alpha, beta)
            
            value = max(value, child_value)
            
            if value >= beta:
                return value  # Beta cutoff
            
            alpha = max(alpha, value)
        
        return value
    
    def _min_value(self, game, depth: int, alpha: float, beta: float) -> float:
        """Minimizing player in minimax with alpha-beta pruning."""
        self.nodes_visited += 1
        
        # Check if we've reached the maximum depth or a terminal state
        if (self.max_depth is not None and depth >= self.max_depth) or game.is_terminal():
            return self._evaluate(game)
        
        value = float('inf')
        valid_actions = game.get_valid_actions()
        
        for action in valid_actions:
            game_copy = self._copy_game(game)
            _, _, done, info = game_copy.step(action)
            
            if done:
                if info["winner"] == self.opponent_id:
                    child_value = -1.0  # Loss
                elif info["winner"] == 0:
                    child_value = 0  # Draw
                else:
                    child_value = 1.0  # Win
            else:
                child_value = self._max_value(game_copy, depth + 1, alpha, beta)
            
            value = min(value, child_value)
            
            if value <= alpha:
                return value  # Alpha cutoff
            
            beta = min(beta, value)
        
        return value
    
    def _max_value_no_pruning(self, game, depth: int) -> float:
        """Maximizing player in minimax without alpha-beta pruning."""
        self.nodes_visited += 1
        
        # Check if we've reached the maximum depth or a terminal state
        if (self.max_depth is not None and depth >= self.max_depth) or game.is_terminal():
            return self._evaluate(game)
        
        value = float('-inf')
        valid_actions = game.get_valid_actions()
        
        for action in valid_actions:
            game_copy = self._copy_game(game)
            _, _, done, info = game_copy.step(action)
            
            if done:
                if info["winner"] == self.player_id:
                    child_value = 1.0  # Win
                elif info["winner"] == 0:
                    child_value = 0  # Draw
                else:
                    child_value = -1.0  # Loss
            else:
                child_value = self._min_value_no_pruning(game_copy, depth + 1)
            
            value = max(value, child_value)
        
        return value
    
    def _min_value_no_pruning(self, game, depth: int) -> float:
        """Minimizing player in minimax without alpha-beta pruning."""
        self.nodes_visited += 1
        
        # Check if we've reached the maximum depth or a terminal state
        if (self.max_depth is not None and depth >= self.max_depth) or game.is_terminal():
            return self._evaluate(game)
        
        value = float('inf')
        valid_actions = game.get_valid_actions()
        
        for action in valid_actions:
            game_copy = self._copy_game(game)
            _, _, done, info = game_copy.step(action)
            
            if done:
                if info["winner"] == self.opponent_id:
                    child_value = -1.0  # Loss
                elif info["winner"] == 0:
                    child_value = 0  # Draw
                else:
                    child_value = 1.0  # Win
            else:
                child_value = self._max_value_no_pruning(game_copy, depth + 1)
            
            value = min(value, child_value)
        
        return value
    
    def _evaluate(self, game) -> float:
        """
        Evaluate the current game state.
        
        For terminal states:
        - Win: +1.0
        - Loss: -1.0
        - Draw: 0.0
        
        For non-terminal states (when using depth-limited search):
        - Heuristic evaluation based on potential winning lines
        """
        # If the game is over, return the actual outcome
        if game.is_terminal():
            winner = game.get_winner()
            if winner == self.player_id:
                return 1.0  # Win
            elif winner == self.opponent_id:
                return -1.0  # Loss
            else:
                return 0.0  # Draw
        
        # For depth-limited search, use a heuristic evaluation
        # Check if game is Connect4 or TicTacToe based on attributes
        if hasattr(game, 'rows') and hasattr(game, 'cols'):
            # Connect4 heuristic
            return self._evaluate_connect4(game)
        else:
            # TicTacToe heuristic
            return self._evaluate_tictactoe(game)
    
    def _evaluate_connect4(self, game) -> float:
        """
        Heuristic evaluation for Connect4.
        Counts potential winning lines for both players.
        """
        board = game.board
        score = 0
        
        # Check horizontal, vertical, and both diagonals for potential wins
        # Give higher scores to positions with more of the player's pieces in a row
        
        # Horizontal
        for row in range(game.rows):
            for col in range(game.cols - 3):
                window = board[row, col:col+4]
                score += self._evaluate_window(window)
        
        # Vertical
        for col in range(game.cols):
            for row in range(game.rows - 3):
                window = board[row:row+4, col]
                score += self._evaluate_window(window)
        
        # Diagonal (positive slope)
        for row in range(game.rows - 3):
            for col in range(game.cols - 3):
                window = [board[row+i, col+i] for i in range(4)]
                score += self._evaluate_window(window)
        
        # Diagonal (negative slope)
        for row in range(3, game.rows):
            for col in range(game.cols - 3):
                window = [board[row-i, col+i] for i in range(4)]
                score += self._evaluate_window(window)
        
        return score
    
    def _evaluate_window(self, window) -> float:
        """Evaluate a window of 4 positions."""
        player_count = np.sum(window == self.player_id)
        opponent_count = np.sum(window == self.opponent_id)
        empty_count = np.sum(window == 0)
        
        # If there's a mix of both players' pieces, this window isn't winnable
        if player_count > 0 and opponent_count > 0:
            return 0
        
        # Score based on how many of player's pieces are in the window
        if player_count > 0:
            if player_count == 3 and empty_count == 1:
                return 0.8  # Near win
            elif player_count == 2 and empty_count == 2:
                return 0.3
            elif player_count == 1 and empty_count == 3:
                return 0.1
        
        # Score based on how many of opponent's pieces are in the window
        if opponent_count > 0:
            if opponent_count == 3 and empty_count == 1:
                return -0.8  # Near loss
            elif opponent_count == 2 and empty_count == 2:
                return -0.3
            elif opponent_count == 1 and empty_count == 3:
                return -0.1
        
        return 0
    
    def _evaluate_tictactoe(self, game) -> float:
        """
        Heuristic evaluation for TicTacToe.
        Simpler than Connect4 since the state space is smaller.
        """
        board = game.board
        score = 0
        
        # Check rows, columns, and diagonals for potential wins
        
        # Rows
        for row in range(3):
            score += self._evaluate_line(board[row, :])
        
        # Columns
        for col in range(3):
            score += self._evaluate_line(board[:, col])
        
        # Diagonals
        score += self._evaluate_line(np.array([board[0, 0], board[1, 1], board[2, 2]]))
        score += self._evaluate_line(np.array([board[0, 2], board[1, 1], board[2, 0]]))
        
        return score
    
    def _evaluate_line(self, line) -> float:
        """Evaluate a line of 3 positions for TicTacToe."""
        player_count = np.sum(line == self.player_id)
        opponent_count = np.sum(line == self.opponent_id)
        empty_count = np.sum(line == 0)
        
        # If both players have pieces in this line, it can't be won
        if player_count > 0 and opponent_count > 0:
            return 0
        
        # Score based on how many of player's pieces are in the line
        if player_count > 0:
            if player_count == 2 and empty_count == 1:
                return 0.6  # Near win
            elif player_count == 1 and empty_count == 2:
                return 0.2
        
        # Score based on how many of opponent's pieces are in the line
        if opponent_count > 0:
            if opponent_count == 2 and empty_count == 1:
                return -0.6  # Near loss
            elif opponent_count == 1 and empty_count == 2:
                return -0.2
        
        return 0
    
    def _copy_game(self, game):
        """Create a deep copy of the game to simulate moves."""
        if hasattr(game, 'rows') and hasattr(game, 'cols'):
            # This is a Connect4 game
            game_copy = type(game)()
            game_copy.board = game.board.copy()
            game_copy.current_player = game.current_player
            game_copy.done = game.done
            game_copy.winner = game.winner
            game_copy.column_heights = game.column_heights.copy()
            return game_copy
        else:
            # This is a TicTacToe game
            game_copy = type(game)()
            game_copy.board = game.board.copy()
            game_copy.current_player = game.current_player
            game_copy.done = game.done
            game_copy.winner = game.winner
            game_copy.valid_actions = game.valid_actions.copy() if hasattr(game, 'valid_actions') else []
            return game_copy