import numpy as np
from typing import Tuple, List, Optional, Dict, Any

class Connect4:
    """
    Implementation of the Connect 4 game.
    
    The board is represented as a 6x7 numpy array with:
    0 - empty space
    1 - player 1 (Red)
    2 - player 2 (Yellow)
    
    The board is indexed as:
    [0,0] [0,1] [0,2] [0,3] [0,4] [0,5] [0,6]
    [1,0] [1,1] [1,2] [1,3] [1,4] [1,5] [1,6]
    [2,0] [2,1] [2,2] [2,3] [2,4] [2,5] [2,6]
    [3,0] [3,1] [3,2] [3,3] [3,4] [3,5] [3,6]
    [4,0] [4,1] [4,2] [4,3] [4,4] [4,5] [4,6]
    [5,0] [5,1] [5,2] [5,3] [5,4] [5,5] [5,6]
    
    With [0,0] being the top-left and [5,6] being the bottom-right.
    Due to gravity, pieces fall to the lowest available position in a column.
    """
    
    def __init__(self):
        # Initialize an empty 6x7 board (rows x columns)
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        # Player 1 starts (Red)
        self.current_player = 1
        # Game is not done yet
        self.done = False
        # No winner yet
        self.winner = None
        # Column heights (number of pieces in each column)
        self.column_heights = np.zeros(self.cols, dtype=int)
        
    def reset(self) -> np.ndarray:
        """Reset the game to its initial state and return the initial board."""
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        self.column_heights = np.zeros(self.cols, dtype=int)
        return self.board.copy()
    
    def get_state(self) -> np.ndarray:
        """Return the current state of the board."""
        return self.board.copy()
    
    def get_valid_actions(self) -> List[int]:
        """Return a list of valid actions (columns that aren't full)."""
        return [col for col in range(self.cols) if self.column_heights[col] < self.rows]
    
    def is_valid_action(self, action: int) -> bool:
        """Check if the action (column) is valid."""
        if action < 0 or action >= self.cols:
            return False
        return self.column_heights[action] < self.rows
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take an action and return the new state, reward, done flag, and info dict.
        
        Args:
            action: The column to drop the piece into (0-6)
            
        Returns:
            state: The new state of the board
            reward: The reward for the action (+1 for win, 0 for draw, -1 for invalid move)
            done: Whether the game is over
            info: Additional information
        """
        # Check if the action is valid
        if not self.is_valid_action(action):
            return self.board.copy(), -1, self.done, {"valid": False, "winner": self.winner}
        
        # Calculate the row where the piece will land
        row = self.rows - 1 - self.column_heights[action]
        
        # Place the piece
        self.board[row, action] = self.current_player
        
        # Update column height
        self.column_heights[action] += 1
        
        # Check if the game is over
        self._check_game_over(row, action)
        
        # Determine reward
        reward = 0
        if self.done:
            if self.winner == self.current_player:
                reward = 1
            # Draw has reward 0
        
        # Switch player
        self.current_player = 3 - self.current_player  # Toggles between 1 and 2
        
        return self.board.copy(), reward, self.done, {"valid": True, "winner": self.winner}
    
    def _check_game_over(self, row: int, col: int) -> None:
        """
        Check if the game is over after placing a piece at (row, col).
        Only checks lines that include the newly placed piece for efficiency.
        """
        player = self.board[row, col]
        
        # Check horizontal
        self._check_line(row, 0, 0, 1, player)
        if self.done:
            return
            
        # Check vertical
        self._check_line(0, col, 1, 0, player)
        if self.done:
            return
            
        # Check diagonal (top-left to bottom-right)
        # Find the top-left starting point of the diagonal
        start_row = row
        start_col = col
        while start_row > 0 and start_col > 0:
            start_row -= 1
            start_col -= 1
        self._check_line(start_row, start_col, 1, 1, player)
        if self.done:
            return
            
        # Check diagonal (top-right to bottom-left)
        # Find the top-right starting point of the diagonal
        start_row = row
        start_col = col
        while start_row > 0 and start_col < self.cols - 1:
            start_row -= 1
            start_col += 1
        self._check_line(start_row, start_col, 1, -1, player)
        if self.done:
            return
            
        # Check for draw (board is full)
        if np.all(self.column_heights == self.rows):
            self.done = True
            self.winner = 0  # 0 indicates a draw
    
    def _check_line(self, start_row: int, start_col: int, row_step: int, col_step: int, player: int) -> None:
        """
        Check if there's a connect-4 starting from (start_row, start_col) and moving in the
        direction specified by (row_step, col_step).
        """
        count = 0
        row, col = start_row, start_col
        
        while 0 <= row < self.rows and 0 <= col < self.cols:
            if self.board[row, col] == player:
                count += 1
                if count == 4:
                    self.done = True
                    self.winner = player
                    return
            else:
                count = 0
                
            row += row_step
            col += col_step
    
    def is_terminal(self) -> bool:
        """Return whether the game is over."""
        return self.done
    
    def get_winner(self) -> Optional[int]:
        """Return the winner of the game (0 for draw, None if game not over)."""
        return self.winner
    
    def render(self) -> str:
        """Render the current board state as a string."""
        symbols = {0: '.', 1: 'R', 2: 'Y'}
        board_str = ""
        
        # Column numbers
        board_str += " "
        for j in range(self.cols):
            board_str += f" {j}"
        board_str += "\n"
        
        # Board
        for i in range(self.rows):
            board_str += "|"
            for j in range(self.cols):
                board_str += f"{symbols[self.board[i, j]]}|"
            board_str += "\n"
            
        # Bottom border
        board_str += "+"
        for j in range(self.cols):
            board_str += "-+"
        board_str += "\n"
                
        return board_str
    
    def get_current_player(self) -> int:
        """Return the current player (1 or 2)."""
        return self.current_player
    
    def __str__(self) -> str:
        """String representation of the board."""
        return self.render()