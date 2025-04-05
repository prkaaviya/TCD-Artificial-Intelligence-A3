import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

class Tournament:
    """
    Tournament class for evaluating game-playing agents.
    
    This class runs multiple games between agents and collects statistics
    on their performance against each other.
    """
    
    def __init__(self, game_class, num_games: int = 100):
        """
        Initialize the tournament.
        
        Args:
            game_class: The class of the game to be played
            num_games: Number of games to play in the tournament
        """
        self.game_class = game_class
        self.num_games = num_games
        self.results = {}
        self.game_logs = []
        
    def run_match(self, agent1, agent2, render: bool = False, 
                    training: bool = False, track_metrics: bool = True) -> Dict[str, Any]:
        """
        Run a single match between two agents.
        
        Args:
            agent1: The first agent (player 1)
            agent2: The second agent (player 2)
            render: Whether to render the game
            training: Whether agents are in training mode
            track_metrics: Whether to track and return metrics
            
        Returns:
            Dictionary with match results and metrics
        """
        game = self.game_class()
        state = game.reset()
        
        # Initialize metrics
        metrics = {
            "moves": 0,
            "winner": None,
            "game_time": 0,
            "agent1_time": 0,
            "agent2_time": 0,
            "agent1_nodes": 0 if hasattr(agent1, 'nodes_visited') else None,
            "agent2_nodes": 0 if hasattr(agent2, 'nodes_visited') else None,
        }
        
        start_time = time.time()
        
        # Main game loop
        while not game.is_terminal():
            current_player = game.get_current_player()
            
            if current_player == 1:
                # Player 1's turn
                agent_start_time = time.time()
                action = agent1.select_action(game, training=training)
                metrics["agent1_time"] += time.time() - agent_start_time
                
                # Track nodes visited for Minimax agents
                if hasattr(agent1, 'nodes_visited'):
                    metrics["agent1_nodes"] += agent1.nodes_visited
                
                # Take the action
                next_state, reward, done, info = game.step(action)
                
                # Learn from the action if in training mode
                if training and hasattr(agent1, 'learn'):
                    agent1.learn(game, reward, done)
                
            else:
                # Player 2's turn
                agent_start_time = time.time()
                action = agent2.select_action(game, training=training)
                metrics["agent2_time"] += time.time() - agent_start_time
                
                # Track nodes visited for Minimax agents
                if hasattr(agent2, 'nodes_visited'):
                    metrics["agent2_nodes"] += agent2.nodes_visited
                
                # Take the action
                next_state, reward, done, info = game.step(action)
                
                # Learn from the action if in training mode
                if training and hasattr(agent2, 'learn'):
                    agent2.learn(game, reward, done)
            
            # Render the game if requested
            if render:
                print(game.render())
                print(f"Player {current_player} chose action: {action}")
                if done:
                    winner = info["winner"]
                    if winner == 0:
                        print("Game ended in a draw.")
                    else:
                        print(f"Player {winner} won!")
            
            # Update metrics
            metrics["moves"] += 1
        
        # Record final game state
        metrics["winner"] = game.get_winner()
        metrics["game_time"] = time.time() - start_time
        
        return metrics
    
    def run_tournament(self, agents_list: List[Tuple[Any, Any]], symmetric: bool = True,
                        render_final: bool = False, training: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Run a tournament between multiple agents.
        
        Args:
            agents_list: List of (agent, name) tuples
            symmetric: Whether to play symmetric matches (A vs B and B vs A)
            render_final: Whether to render the final game
            training: Whether agents are in training mode
            
        Returns:
            Dictionary with tournament results
        """
        results = {}
        
        # Initialize results structure
        for agent1, name1 in agents_list:
            if name1 not in results:
                results[name1] = {
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "win_rate": 0.0,
                    "avg_moves": 0.0,
                    "avg_time": 0.0,
                    "total_games": 0
                }
        
        # Run all matchups
        matchups = []
        for i, (agent1, name1) in enumerate(agents_list):
            for j, (agent2, name2) in enumerate(agents_list):
                if i == j:  # Skip self-play
                    continue
                if not symmetric and j < i:  # Skip redundant matchups if not symmetric
                    continue
                matchups.append((agent1, name1, agent2, name2))
        
        # Show progress bar for tournament
        for agent1, name1, agent2, name2 in tqdm(matchups, desc="Running tournament"):
            print(f"\n\n\nMatchup: {name1} vs {name2}")
            
            # Clone agents to ensure independence for each matchup
            agent1_clone = agent1.__class__(agent1.player_id)
            agent2_clone = agent2.__class__(agent2.player_id)
            
            # Try to copy important attributes
            for attr in ['use_alpha_beta', 'max_depth', 'q_table']:
                if hasattr(agent1, attr):
                    setattr(agent1_clone, attr, getattr(agent1, attr))
                if hasattr(agent2, attr):
                    setattr(agent2_clone, attr, getattr(agent2, attr))
            
            match_results = []
            
            # Run multiple games
            for game_idx in tqdm(range(self.num_games), desc=f"Games ({name1} vs {name2})"):
                render = render_final and game_idx == self.num_games - 1
                match_result = self.run_match(agent1_clone, agent2_clone, render=render, training=training)
                match_results.append(match_result)
            
            # Compute statistics
            wins = sum(1 for r in match_results if r["winner"] == 1)
            losses = sum(1 for r in match_results if r["winner"] == 2)
            draws = sum(1 for r in match_results if r["winner"] == 0)
            
            avg_moves = sum(r["moves"] for r in match_results) / len(match_results)
            avg_time = sum(r["game_time"] for r in match_results) / len(match_results)
            
            # Update agent1's results
            results[name1]["wins"] += wins
            results[name1]["losses"] += losses
            results[name1]["draws"] += draws
            results[name1]["total_games"] += len(match_results)
            
            # Update agent2's results
            results[name2]["wins"] += losses  # Agent2's wins are agent1's losses
            results[name2]["losses"] += wins  # Agent2's losses are agent1's wins
            results[name2]["draws"] += draws
            results[name2]["total_games"] += len(match_results)
            
            # Store detailed matchup results
            matchup_key = f"{name1} vs {name2}"
            results[matchup_key] = {
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": wins / len(match_results),
                "draw_rate": draws / len(match_results),
                "avg_moves": avg_moves,
                "avg_time": avg_time,
                "avg_agent1_time": sum(r["agent1_time"] for r in match_results) / len(match_results),
                "avg_agent2_time": sum(r["agent2_time"] for r in match_results) / len(match_results),
            }
            
            # For Minimax agents, track node statistics
            if any(r["agent1_nodes"] is not None for r in match_results):
                results[matchup_key]["avg_agent1_nodes"] = sum(r["agent1_nodes"] for r in match_results 
                                                            if r["agent1_nodes"] is not None) / len(match_results)
            
            if any(r["agent2_nodes"] is not None for r in match_results):
                results[matchup_key]["avg_agent2_nodes"] = sum(r["agent2_nodes"] for r in match_results 
                                                            if r["agent2_nodes"] is not None) / len(match_results)
            
            print(f"Results: {wins} wins, {losses} losses, {draws} draws")
            print(f"Win rate: {wins / len(match_results):.2f}")
            print(f"Average moves per game: {avg_moves:.2f}")
        
        # Calculate final statistics
        for name in [name for agent, name in agents_list]:
            if results[name]["total_games"] > 0:
                results[name]["win_rate"] = results[name]["wins"] / results[name]["total_games"]
                results[name]["draw_rate"] = results[name]["draws"] / results[name]["total_games"]
        
        self.results = results
        return results
    
    def save_results_to_csv(self, path: str) -> None:
        """
        Save tournament results to a CSV file.
        
        Args:
            path: Path to save the CSV file
        """
        if not self.results:
            print("No results to save. Run a tournament first.")
            return
        
        # Extract agent statistics
        agent_stats = {}
        for key, value in self.results.items():
            if " vs " not in key:  # Agent summary
                agent_stats[key] = value
        
        # Convert to DataFrame
        df = pd.DataFrame(agent_stats).T
        df.index.name = 'Agent'
        
        # Save to CSV
        df.to_csv(path)
        print(f"Results saved to {path}")
    
    def print_detailed_results(self) -> None:
        """Print detailed results of the tournament."""
        if not self.results:
            print("No results to print. Run a tournament first.")
            return
        
        print("\n===== TOURNAMENT RESULTS =====\n")
        
        # Print overall agent statistics
        print("Overall Agent Performance:")
        agent_stats = {}
        for key, value in self.results.items():
            if " vs " not in key:  # Agent summary
                agent_stats[key] = value
        
        df_overall = pd.DataFrame(agent_stats).T
        df_overall['win_rate'] = df_overall['win_rate'].apply(lambda x: f"{x:.2%}")
        df_overall['draw_rate'] = df_overall.get('draw_rate', 0).apply(lambda x: f"{x:.2%}")
        print(df_overall[['wins', 'draws', 'losses', 'win_rate', 'total_games']])
        
        print("\nDetailed Matchup Results:")
        # Print each matchup result
        for key, value in self.results.items():
            if " vs " in key:  # Matchup result
                print(f"\n{key}:")
                print(f"  Wins: {value['wins']}, Losses: {value['losses']}, Draws: {value['draws']}")
                print(f"  Win Rate: {value['win_rate']:.2%}, Draw Rate: {value['draw_rate']:.2%}")
                print(f"  Average Moves: {value['avg_moves']:.2f}")
                print(f"  Average Game Time: {value['avg_time']:.4f}s")
                print(f"  Player 1 Average Time: {value['avg_agent1_time']:.4f}s")
                print(f"  Player 2 Average Time: {value['avg_agent2_time']:.4f}s")
                
                if 'avg_agent1_nodes' in value:
                    print(f"  Player 1 Average Nodes Visited: {value['avg_agent1_nodes']:.1f}")
                if 'avg_agent2_nodes' in value:
                    print(f"  Player 2 Average Nodes Visited: {value['avg_agent2_nodes']:.1f}")
