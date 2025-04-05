import numpy as np
import random
import pickle
from typing import Tuple, Dict, List, Union, Any
import os

class QLearningAgent:
    """
    Q-learning agent for playing games.
    
    This agent uses tabular Q-learning to learn a policy.
    State representation is simplified to make learning tractable.
    """
    
    def __init__(self, player_id: int, learning_rate: float = 0.1, discount_factor: float = 0.9, 
                    exploration_rate: float = 0.3, exploration_decay: float = 0.995, min_exploration: float = 0.01,
                    load_qtable: bool = False, save_path: str = None):
        """
        Initialize the Q-learning agent.
        
        Args:
            player_id: The ID of the player (1 or 2)
            learning_rate: Alpha - learning rate
            discount_factor: Gamma - discount factor for future rewards
            exploration_rate: Epsilon - exploration rate
            exploration_decay: Factor to decay exploration rate after each episode
            min_exploration: Minimum exploration rate
            load_qtable: Whether to load an existing Q-table
            save_path: Path to save/load Q-table
        """
        self.player_id = player_id
        self.opponent_id = 3 - player_id
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        self.q_table = {}
        self.name = "Q-Learning Agent"
        self.save_path = save_path or f"q_table_player{player_id}.pkl"
        
        # Keep track of the current game for learning
        self.current_state = None
        self.current_action = None
        
        # Load Q-table if specified
        if load_qtable and os.path.exists(self.save_path):
            self.load_q_table()
    
    def state_to_tuple(self, game) -> tuple:
        """
        Convert a game state to a tuple that can be used as a dictionary key.
        Uses a simplified representation to keep the state space manageable.
        
        Args:
            game: The game object
            
        Returns:
            A tuple representation of the state
        """
        if hasattr(game, 'rows') and hasattr(game, 'cols'):
            # This is a Connect4 game - use a more compact representation
            # For Connect4, we can't use the raw board as a key because it's too large
            # Instead, create a tuple of tuples where each inner tuple represents a column
            columns = []
            for col in range(game.cols):
                column_pieces = []
                for row in range(game.rows):
                    if game.board[row, col] != 0:
                        # Store (row, player_id) for each piece in the column
                        column_pieces.append((row, int(game.board[row, col])))
                columns.append(tuple(column_pieces))
            
            # Include the current player in the state
            return (tuple(columns), game.current_player)
        else:
            # This is a TicTacToe game - we can use the flattened board
            # Convert board to tuple of tuples for hashability
            board_tuple = tuple(tuple(row) for row in game.board)
            return (board_tuple, game.current_player)
    
    def get_q_value(self, state: tuple, action: Union[Tuple[int, int], int]) -> float:
        """
        Get the Q-value for a state-action pair.
        
        Args:
            state: The state tuple
            action: The action
            
        Returns:
            The Q-value
        """
        if state not in self.q_table:
            self.q_table[state] = {}
        
        # Convert action to a hashable type if it's not already
        if isinstance(action, np.ndarray):
            action = tuple(action)
        
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
            
        return self.q_table[state][action]
    
    def update_q_value(self, state: tuple, action: Union[Tuple[int, int], int], 
                        reward: float, next_state: tuple, done: bool) -> None:
        """
        Update the Q-value for a state-action pair.
        
        Args:
            state: The current state tuple
            action: The action taken
            reward: The reward received
            next_state: The resulting state tuple
            done: Whether the episode is done
        """
        # Convert action to a hashable type if it's not already
        if isinstance(action, np.ndarray):
            action = tuple(action)
        
        # Get current Q-value
        q_value = self.get_q_value(state, action)
        
        if done:
            # If the episode is done, there is no next state
            max_next_q = 0
        else:
            # Get max Q-value for next state
            if next_state in self.q_table and self.q_table[next_state]:
                max_next_q = max(self.q_table[next_state].values())
            else:
                max_next_q = 0
        
        # Q-learning update formula
        new_q_value = q_value + self.learning_rate * (reward + self.discount_factor * max_next_q - q_value)
        
        if state not in self.q_table:
            self.q_table[state] = {}
        
        self.q_table[state][action] = new_q_value
    
    def select_action(self, game, training: bool = False) -> Union[Tuple[int, int], int]:
        """
        Select an action based on the current game state.
        
        Args:
            game: The game object
            training: Whether the agent is in training mode
            
        Returns:
            The selected action
        """
        state = self.state_to_tuple(game)
        valid_actions = game.get_valid_actions()
        
        if not valid_actions:
            raise ValueError("No valid actions available")
        
        # Save current state for learning update later
        if training:
            self.current_state = state
        
        # Exploration: random action
        if training and random.random() < self.exploration_rate:
            action = random.choice(valid_actions)
            if training:
                self.current_action = action
            return action
        
        # Exploitation: best known action
        q_values = {action: self.get_q_value(state, action) for action in valid_actions}
        
        # Find actions with the maximum Q-value
        max_q = max(q_values.values()) if q_values else 0
        best_actions = [action for action, q_value in q_values.items() if q_value == max_q]
        
        # If multiple actions have the same Q-value, choose randomly among them
        action = random.choice(best_actions) if best_actions else random.choice(valid_actions)
        
        if training:
            self.current_action = action
            
        return action
    
    def learn(self, game, reward: float, done: bool) -> None:
        """
        Learn from the most recent action.
        
        Args:
            game: The game object after the action was taken
            reward: The reward received
            done: Whether the episode is done
        """
        if self.current_state is None or self.current_action is None:
            return
        
        next_state = self.state_to_tuple(game) if not done else None
        
        self.update_q_value(self.current_state, self.current_action, reward, next_state, done)
        
        # Prepare for next action
        self.current_state = next_state
        self.current_action = None
        
        # Decay exploration rate
        if done:
            self.exploration_rate = max(self.min_exploration, 
                                       self.exploration_rate * self.exploration_decay)
    
    def save_q_table(self, path: str = None) -> None:
        """Save the Q-table to a file."""
        save_path = path or self.save_path
        with open(save_path, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {save_path}")
        print(f"Q-table size: {len(self.q_table)} states")
    
    def load_q_table(self, path: str = None) -> None:
        """Load the Q-table from a file."""
        load_path = path or self.save_path
        if os.path.exists(load_path):
            with open(load_path, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Q-table loaded from {load_path}")
            print(f"Q-table size: {len(self.q_table)} states")
        else:
            print(f"No Q-table found at {load_path}")
    
    def reset(self) -> None:
        """Reset agent state for a new episode."""
        self.current_state = None
        self.current_action = None
