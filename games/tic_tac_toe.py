import numpy as np
from typing import Tuple, List, Optional, Dict, Any

class TicTacToe:
    """
    Implementation of the Tic Tac Toe game.

    The board is represented as a 3x3 numpy array with:
    0 - empty space
    1 - player 1 (X)
    2 - player 2 (O)
    """

    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        self.valid_actions = [(i, j) for i in range(3) for j in range(3)]

    def reset(self) -> np.ndarray:
        """Reset the game to its initial state and return the initial board."""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None
        self.valid_actions = [(i, j) for i in range(3) for j in range(3)]
        return self.board.copy()

    def get_state(self) -> np.ndarray:
        """Return the current state of the board."""
        return self.board.copy()

    def get_valid_actions(self) -> List[Tuple[int, int]]:
        """Return a list of valid actions (empty spaces on the board)."""
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def is_valid_action(self, action: Tuple[int, int]) -> bool:
        """Check if the action is valid."""
        row, col = action
        if row < 0 or row >= 3 or col < 0 or col >= 3:
            return False
        return self.board[row, col] == 0

    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take an action and return the new state, reward, done flag, and info dict.

        Args:
            action: A tuple (row, col) indicating where to place the piece

        Returns:
            state: The new state of the board
            reward: The reward for the action (+1 for win, 0 for draw, -1 for invalid move)
            done: Whether the game is over
            info: Additional information
        """
        row, col = action

        if not self.is_valid_action(action):
            return self.board.copy(), -1, self.done, {"valid": False, "winner": self.winner}

        self.board[row, col] = self.current_player

        if action in self.valid_actions:
            self.valid_actions.remove(action)

        self._check_game_over()

        reward = 0
        if self.done:
            if self.winner == self.current_player:
                reward = 1
            # draw has reward 0

        self.current_player = 3 - self.current_player  # toggles between 1 and 2

        return self.board.copy(), reward, self.done, {"valid": True, "winner": self.winner}

    def _check_game_over(self) -> None:
        """Check if the game is over (win or draw)."""
        for row in range(3):
            if self.board[row, 0] == self.board[row, 1] == self.board[row, 2] != 0:
                self.done = True
                self.winner = self.board[row, 0]
                return

        for col in range(3):
            if self.board[0, col] == self.board[1, col] == self.board[2, col] != 0:
                self.done = True
                self.winner = self.board[0, col]
                return

        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] != 0:
            self.done = True
            self.winner = self.board[0, 0]
            return

        if self.board[0, 2] == self.board[1, 1] == self.board[2, 0] != 0:
            self.done = True
            self.winner = self.board[0, 2]
            return

        if np.all(self.board != 0):
            self.done = True
            self.winner = 0  # 0 indicates a draw
            return

    def is_terminal(self) -> bool:
        """Return whether the game is over."""
        return self.done

    def get_winner(self) -> Optional[int]:
        """Return the winner of the game (0 for draw, None if game not over)."""
        return self.winner

    def render(self) -> str:
        """Render the current board state as a string."""
        symbols = {0: ' ', 1: 'X', 2: 'O'}
        board_str = ""

        board_str += "  0 1 2\n"
        for i in range(3):
            board_str += f"{i} "
            for j in range(3):
                board_str += symbols[self.board[i, j]]
                if j < 2:
                    board_str += "|"
            board_str += "\n"
            if i < 2:
                board_str += "  -+-+-\n"

        return board_str

    def get_current_player(self) -> int:
        """Return the current player (1 or 2)."""
        return self.current_player

    def __str__(self) -> str:
        """String representation of the board."""
        return self.render()
