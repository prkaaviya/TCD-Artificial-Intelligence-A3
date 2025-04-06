"""
Main entry point for running games - Tic Tac Toe & Connect 4.
"""
import argparse
import time
import os

from games.tic_tac_toe import TicTacToe
from games.connect4 import Connect4

from agents.default_agent import DefaultAgent
from agents.minimax_agent import MinimaxAgent
from agents.qlearning_agent import QLearningAgent

from evaluator.evaluator import Evaluator

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

    agent = QLearningAgent(player_id=1, exploration_rate=0.3, save_path=save_path)

    game = game_class()

    # start training loop
    for episode in range(num_episodes):
        game.reset()
        agent.reset()

        # print progress every 1000 episodes
        if episode % 1000 == 0:
            print(f"Episode {episode}/{num_episodes}, \
                    exploration rate: {agent.exploration_rate:.4f}")

        done = False

        while not done:
            current_player = game.get_current_player()

            if current_player == 1:  # the Q-learning agent's turn
                action = agent.select_action(game, training=True)
                next_state, reward, done, info = game.step(action)
                agent.learn(game, reward, done)
            else:  # the Opponent's turn
                action = opponent.select_action(game)
                next_state, reward, done, info = game.step(action)

        # adjust reward based on the game outcome
        winner = info.get("winner", None)
        if winner == 1:  # agent won
            agent.learn(game, 1.0, True)
        elif winner == 2:  # agent lost
            agent.learn(game, -1.0, True)
        else:  # game ended in draw
            agent.learn(game, 0.2, True)  # give small positive reward for a draw

    if save_path:
        agent.save_q_table(save_path)

    return agent

def play_single_game(game_class, agent1, agent2, render=True, max_time=1800):
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
    move_count = 0
    max_moves = 9 if game_class == TicTacToe else 42

    start_time = time.time()
    stats = {
        "agent1_nodes": 0,
        "agent2_nodes": 0,
        "agent1_time": 0,
        "agent2_time": 0,
        "moves_completed": 0
    }

    if render:
        print(game.render())

    while not done:
        current_player = game.get_current_player()
        current_agent = agent1 if current_player == 1 else agent2

        # check if we've exceeded the time limit
        if time.time() - start_time > max_time:
            print(f"\nTime limit of {max_time} seconds exceeded!")
            print(f"\nCompleted {move_count} moves out of maximum {max_moves} possible moves.")
            break

        # track time and nodes for current move
        move_start_time = time.time()

        # get the initial node count if available
        initial_nodes = 0
        if hasattr(current_agent, 'nodes_visited'):
            initial_nodes = current_agent.nodes_visited

        action = current_agent.select_action(game)

        # calculate time spent and nodes visited
        move_time = time.time() - move_start_time
        if current_player == 1:
            stats["agent1_time"] += move_time
            if hasattr(agent1, 'nodes_visited'):
                nodes_this_move = abs(agent1.nodes_visited - initial_nodes)
                stats["agent1_nodes"] += nodes_this_move
                print(f"Move {move_count+1}: Player 1 explored {nodes_this_move} nodes in {move_time:.4f}s")
        else:
            stats["agent2_time"] += move_time
            if hasattr(agent2, 'nodes_visited'):
                nodes_this_move = abs(agent2.nodes_visited - initial_nodes)
                stats["agent2_nodes"] += nodes_this_move
                print(f"Move {move_count+1}: Player 2 explored {nodes_this_move} nodes in {move_time:.4f}s")

        _, _, done, info = game.step(action)

        move_count += 1
        stats["moves_completed"] = move_count

        if render:
            print(f"\nPlayer {current_player} ({current_agent.name}) chose action: {action}")
            print(game.render())

    winner = info.get("winner", None)
    if winner == 0:
        print("\n\nGame ended in a draw!")
    else:
        print(f"\n\nPlayer {winner} ({agent1.name if winner == 1 else agent2.name}) won!")

    total_time = time.time() - start_time
    print("\n*** Single Game Stats ***")
    print(f"\nTotal moves completed: {move_count}/{max_moves}")
    print(f"Total game time: {total_time:.2f}s")

    if hasattr(agent1, 'nodes_visited'):
        print(f"Player 1 ({agent1.name}) explored {stats['agent1_nodes']} nodes in {stats['agent1_time']:.2f}s.")
    if hasattr(agent2, 'nodes_visited'):
        print(f"Player 2 ({agent2.name}) explored {stats['agent2_nodes']} nodes in {stats['agent2_time']:.2f}s.")

    return stats

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

        e = Evaluator(game_class, num_games=num_games)
        e_results = e.run_game(minimax_agent, default_agent, render=False)

        results[depth] = {
            "nodes_visited": e_results["agent1_nodes"],
            "execution_time": e_results["agent1_time"],
            "depth": depth,
            "result": "Win" if e_results["winner"] == 1 else 
                    "Draw" if e_results["winner"] == 0 else "Loss"
        }

        print(f"  Depth {depth_name}: Nodes visited = {e_results['agent1_nodes']}, Time = {e_results['agent1_time']:.4f}s")
        print(f"  Result: {results[depth]['result']}")

    return results

def main():
    """Main function to run the game-playing agents comparison."""
    parser = argparse.ArgumentParser(description='Compare game-playing agents.')
    parser.add_argument('--game', type=str, default='tictactoe', choices=['tictactoe', 'connect4'],
                        help='Game to play (tictactoe or connect4)')
    parser.add_argument('--mode', type=str, default='evaluator',
                        choices=['evaluator', 'train', 'play', 'depth_test'],
                        help='Mode to run (evaluator, train, play, depth_test)')
    parser.add_argument('--num_games', type=int, default=100,
                        help='Number of games to play in evaluator mode')
    parser.add_argument('--render', action='store_true',
                        help='Render games in play mode')
    parser.add_argument('--train_episodes', type=int, default=5000,
                        help='Number of episodes for Q-learning training')
    parser.add_argument('--load_qtable', action='store_true',
                        help='Load previously trained Q-table for Q-learning')

    args = parser.parse_args()

    game_class = TicTacToe if args.game == 'tictactoe' else Connect4
    game_name = args.game.capitalize()

    os.makedirs('results', exist_ok=True)

    if args.mode == 'train':
        opponent = DefaultAgent(player_id=2)
        save_path = f'results/q_table_{args.game}.pkl'

        train_qlearning_agent(game_class, opponent, 
                                num_episodes=args.train_episodes,
                                save_path=save_path)

        print(f"Training completed. Q-table saved to {save_path}")

    elif args.mode == 'play':
        # play a single game
        minimax_agent = MinimaxAgent(player_id=1, use_alpha_beta=True,
                                        max_depth=3 if args.game == 'connect4' else None)
        default_agent = DefaultAgent(player_id=2)

        print(f"\n*** Playing a game of {game_name} ***")
        print(f"\nPlayer 1: {minimax_agent.name}")
        print(f"\nPlayer 2: {default_agent.name}")
        print("\n")

        play_single_game(game_class, minimax_agent, default_agent, render=args.render)

    elif args.mode == 'depth_test':
        # play test the effect of depth on Minimax performance
        depths = [1, 2, 3, 4, 5] if args.game == 'connect4' else [None, 1, 2, 3, 4, 5]
        results = compare_minimax_depth_effect(game_class, depths=depths, num_games=args.num_games)

        print("\nDepth test summary:")
        for depth, data in results.items():
            depth_name = "Unlimited" if depth is None else str(depth)
            print(f"Depth {depth_name}: Nodes = {data['nodes_visited']}, Time = {data['execution_time']:.4f}s, Result = {data['result']}")

    elif args.mode == 'evaluator':
        # run game evaluator between all agents
        print(f"\n*** Running benchmark or game performance evaulator for {game_name} with {args.num_games} games per agents ***")

        agents = []

        default_agent = DefaultAgent(player_id=1)
        agents.append((default_agent, "Default Agent"))

        minimax_depth = 3 if args.game == 'connect4' else None
        minimax_no_ab = MinimaxAgent(player_id=1, use_alpha_beta=False, max_depth=minimax_depth)
        agents.append((minimax_no_ab, "Minimax without Alpha-Beta"))

        minimax_with_ab = MinimaxAgent(player_id=1, use_alpha_beta=True, max_depth=minimax_depth)
        agents.append((minimax_with_ab, "Minimax with Alpha-Beta"))

        qtable_path = f'results/q_table_{args.game}.pkl'
        qlearning_agent = QLearningAgent(player_id=1,
                                        load_qtable=args.load_qtable,
                                        save_path=qtable_path)

        if not os.path.exists(qtable_path) or not args.load_qtable:
            print("\nNo pre-trained Q-table found or load_qtable=False. Training a new Q-learning agent...")
            qlearning_agent = train_qlearning_agent(game_class, default_agent,
                                                    num_episodes=args.train_episodes,
                                                    save_path=qtable_path)

        agents.append((qlearning_agent, "Q-Learning Agent"))

        e = Evaluator(game_class, num_games=args.num_games)
        results = e.run_evaluator(agents, symmetric=True, render_final=args.render)

        e.print_detailed_results()
        e.save_results_to_csv(f"results/{args.game}_evaluator_results.csv")

if __name__ == "__main__":
    main()
