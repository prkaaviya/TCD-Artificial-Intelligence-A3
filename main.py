import argparse
import numpy as np
import time
import os
from typing import List, Dict, Any, Tuple

# Import game environments
from games.tic_tac_toe import TicTacToe
from games.connect4 import Connect4

# Import agents
from agents.default_agent import DefaultAgent
from agents.minimax_agent import MinimaxAgent
from agents.qlearning_agent import QLearningAgent

# Import tournament
from evaluation.tournament import Tournament

def train_qlearning_agent(game_class, opponent, num_episodes=10000, save_path=None):
    """
    Train a Q-learning agent against the specified opponent.
    
    Args:
        game_class: The game class to use
        opponent: The opponent agent
        num_episodes: Number of training episodes
        save_path: Path to save the trained agent
        
    Returns:
        The trained Q-learning agent
    """
    print(f"Training Q-learning agent for {game_class.__name__} over {num_episodes} episodes...")
    
    # Create the Q-learning agent
    agent = QLearningAgent(player_id=1, exploration_rate=0.3, save_path=save_path)
    
    # Create a game instance
    game = game_class()
    
    # Training loop
    for episode in range(num_episodes):
        game.reset()
        agent.reset()
        
        # Print progress every 1000 episodes
        if episode % 1000 == 0:
            print(f"Episode {episode}/{num_episodes}, exploration rate: {agent.exploration_rate:.4f}")
        
        done = False
        
        while not done:
            # Get the current player
            current_player = game.get_current_player()
            
            if current_player == 1:  # Q-learning agent's turn
                action = agent.select_action(game, training=True)
                next_state, reward, done, info = game.step(action)
                agent.learn(game, reward, done)
            else:  # Opponent's turn
                action = opponent.select_action(game)
                next_state, reward, done, info = game.step(action)
        
        # Adjust reward based on the game outcome
        winner = info.get("winner", None)
        if winner == 1:  # Agent won
            agent.learn(game, 1.0, True)
        elif winner == 2:  # Agent lost
            agent.learn(game, -1.0, True)
        else:  # Draw
            agent.learn(game, 0.2, True)  # Small positive reward for a draw
    
    # Save the trained agent's Q-table
    if save_path:
        agent.save_q_table(save_path)
    
    return agent

def play_single_game(game_class, agent1, agent2, render=True):
    """
    Play a single game between two agents with rendering.
    
    Args:
        game_class: The game class to use
        agent1: The first agent (player 1)
        agent2: The second agent (player 2)
        render: Whether to render the game
    """
    game = game_class()
    game.reset()
    
    done = False
    
    if render:
        print(game.render())
    
    while not done:
        current_player = game.get_current_player()
        current_agent = agent1 if current_player == 1 else agent2
        
        action = current_agent.select_action(game)
        _, _, done, info = game.step(action)
        
        if render:
            print(f"Player {current_player} ({current_agent.name}) chose action: {action}")
            print(game.render())
    
    winner = info.get("winner", None)
    if winner == 0:
        print("Game ended in a draw!")
    else:
        print(f"Player {winner} ({agent1.name if winner == 1 else agent2.name}) won!")

def compare_minimax_depth_effect(game_class, depths=[None, 1, 2, 3, 4, 5], num_games=10):
    """
    Compare the effect of different search depths on Minimax performance.
    
    Args:
        game_class: The game class to use
        depths: List of depths to test
        num_games: Number of games to play for each depth
        
    Returns:
        Dictionary with results
    """
    print(f"Comparing Minimax depth effect for {game_class.__name__}...")
    
    default_agent = DefaultAgent(player_id=2)
    results = {}
    
    for depth in depths:
        depth_name = "Unlimited" if depth is None else str(depth)
        print(f"Testing depth: {depth_name}")
        
        minimax_agent = MinimaxAgent(player_id=1, use_alpha_beta=True, max_depth=depth)
        
        # Create a mini-tournament
        tournament = Tournament(game_class, num_games=num_games)
        tournament_results = tournament.run_match(minimax_agent, default_agent, render=False)
        
        # Store results
        results[depth] = {
            "nodes_visited": tournament_results["agent1_nodes"],
            "execution_time": tournament_results["agent1_time"],
            "depth": depth,
            "result": "Win" if tournament_results["winner"] == 1 else 
                    "Draw" if tournament_results["winner"] == 0 else "Loss"
        }
        
        print(f"  Depth {depth_name}: Nodes visited = {tournament_results['agent1_nodes']}, Time = {tournament_results['agent1_time']:.4f}s")
        print(f"  Result: {results[depth]['result']}")
    
    return results

def main():
    """Main function to run the game-playing agents comparison."""
    parser = argparse.ArgumentParser(description='Compare game-playing agents.')
    parser.add_argument('--game', type=str, default='tictactoe', choices=['tictactoe', 'connect4'],
                        help='Game to play (tictactoe or connect4)')
    parser.add_argument('--mode', type=str, default='tournament', 
                        choices=['tournament', 'train', 'play', 'depth_test'],
                        help='Mode to run (tournament, train, play, depth_test)')
    parser.add_argument('--num_games', type=int, default=100,
                        help='Number of games to play in tournament mode')
    parser.add_argument('--render', action='store_true',
                        help='Render games in play mode')
    parser.add_argument('--train_episodes', type=int, default=5000,
                        help='Number of episodes for Q-learning training')
    parser.add_argument('--load_qtable', action='store_true',
                        help='Load previously trained Q-table for Q-learning')
    
    args = parser.parse_args()
    
    # Select the game
    game_class = TicTacToe if args.game == 'tictactoe' else Connect4
    game_name = args.game.capitalize()
    
    # Create the output directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    if args.mode == 'train':
        # Train a Q-learning agent
        opponent = DefaultAgent(player_id=2)
        save_path = f'results/q_table_{args.game}.pkl'
        
        train_qlearning_agent(game_class, opponent, 
                             num_episodes=args.train_episodes,
                             save_path=save_path)
        
        print(f"Training completed. Q-table saved to {save_path}")
    
    elif args.mode == 'play':
        # Play a single game
        minimax_agent = MinimaxAgent(player_id=1, use_alpha_beta=True, 
                                     max_depth=3 if args.game == 'connect4' else None)
        default_agent = DefaultAgent(player_id=2)
        
        print(f"Playing a game of {game_name}:")
        print(f"Player 1: {minimax_agent.name}")
        print(f"Player 2: {default_agent.name}")
        
        play_single_game(game_class, minimax_agent, default_agent, render=args.render)
    
    elif args.mode == 'depth_test':
        # Test the effect of depth on Minimax performance
        depths = [1, 2, 3, 4, 5] if args.game == 'connect4' else [None, 1, 2, 3, 4, 5]
        results = compare_minimax_depth_effect(game_class, depths=depths, num_games=args.num_games)
        
        # Print summary
        print("\nDepth test summary:")
        for depth, data in results.items():
            depth_name = "Unlimited" if depth is None else str(depth)
            print(f"Depth {depth_name}: Nodes = {data['nodes_visited']}, Time = {data['execution_time']:.4f}s, Result = {data['result']}")
    
    elif args.mode == 'tournament':
        # Run a tournament between all agents
        print(f"Running tournament for {game_name} with {args.num_games} games per matchup...")
        
        # Create agents
        agents = []
        
        # Default agent
        default_agent = DefaultAgent(player_id=1)
        agents.append((default_agent, "Default Agent"))
        
        # Minimax without alpha-beta
        minimax_depth = 3 if args.game == 'connect4' else None
        minimax_no_ab = MinimaxAgent(player_id=1, use_alpha_beta=False, max_depth=minimax_depth)
        agents.append((minimax_no_ab, "Minimax without Alpha-Beta"))
        
        # Minimax with alpha-beta
        minimax_with_ab = MinimaxAgent(player_id=1, use_alpha_beta=True, max_depth=minimax_depth)
        agents.append((minimax_with_ab, "Minimax with Alpha-Beta"))
        
        # Q-learning agent
        qtable_path = f'results/q_table_{args.game}.pkl'
        qlearning_agent = QLearningAgent(player_id=1, load_qtable=args.load_qtable, save_path=qtable_path)
        
        # If Q-table doesn't exist or load_qtable is False, train a new one
        if not os.path.exists(qtable_path) or not args.load_qtable:
            print("No pre-trained Q-table found or load_qtable=False. Training a new Q-learning agent...")
            qlearning_agent = train_qlearning_agent(game_class, default_agent, 
                                                    num_episodes=args.train_episodes,
                                                    save_path=qtable_path)
        
        agents.append((qlearning_agent, "Q-Learning Agent"))
        
        # Create and run the tournament
        tournament = Tournament(game_class, num_games=args.num_games)
        results = tournament.run_tournament(agents, symmetric=True, render_final=args.render)
        
        # Print and save results
        tournament.print_detailed_results()
        tournament.plot_results(title=f"{game_name} Tournament Results",
                                save_path=f"results/{args.game}_tournament_results.png")
        tournament.save_results_to_csv(f"results/{args.game}_tournament_results.csv")

if __name__ == "__main__":
    main()