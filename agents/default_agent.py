import random
import numpy as np
from typing import Tuple, List, Union

class DefaultAgent:
    """
    A default opponent for game playing.
    
    This agent is better than random but still simple:
    1. If there's a winning move, take it
    2. If opponent has a winning move, block it
    3. Otherwise, make a random valid move
    """

    def __init__(self, player_id: int):
        """
        Initialize the agent.

        Args:
            player_id: The ID of the player (1 or 2)
        """
        self.player_id = player_id
        self.opponent_id = 3 - player_id  # if player_id is 1, opponent is 2 and vice versa
        self.name = "Default Agent"

    def select_action(self, game, training: bool = False) -> Union[Tuple[int, int], int]:
        """
        Select an action based on the current game state.
        
        Args:
            game: The game object (TicTacToe or Connect4)
            
        Returns:
            action: For TicTacToe: (row, col) tuple, for Connect4: column index
        """
        valid_actions = game.get_valid_actions()
        if not valid_actions:
            raise ValueError("No valid actions available")

        # check if there's a winning move
        for action in valid_actions:
            # make a copy of the game to simulate moves
            game_copy = self._copy_game(game)

            # simulate taking the action
            state, _, done, info = game_copy.step(action)

            # if the game is over and agent won, take this action
            if done and info["winner"] == self.player_id:
                return action

        # check if the opponent has a winning move and block it
        # simulate opponent's turn first
        for action in valid_actions:
            game_copy = self._copy_game(game)
            
            # change current player to opponent to simulate their move
            game_copy.current_player = self.opponent_id

            # simulate opponent taking the action
            state, _, done, info = game_copy.step(action)

            # if the game would be over and opponent would win, block this action
            if done and info["winner"] == self.opponent_id:
                return action

        # otherwise make a random valid move
        return random.choice(valid_actions)
    
    def _copy_game(self, game):
        """Create a deep copy of the game to simulate moves."""
        if hasattr(game, 'rows') and hasattr(game, 'cols'):
            game_copy = type(game)()
            game_copy.board = game.board.copy()
            game_copy.current_player = game.current_player
            game_copy.done = game.done
            game_copy.winner = game.winner
            game_copy.column_heights = game.column_heights.copy()
            return game_copy
        else:
            game_copy = type(game)()
            game_copy.board = game.board.copy()
            game_copy.current_player = game.current_player
            game_copy.done = game.done
            game_copy.winner = game.winner
            game_copy.valid_actions = game.valid_actions.copy() \
                if hasattr(game, 'valid_actions') else []
            return game_copy
