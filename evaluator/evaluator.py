import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

class Evaluator:
    """
    Evaluator class for evaluating game-playing agents.
    
    This class runs multiple games between agents and collects statistics
    on their performance against each other.
    """

    def __init__(self, game_class, num_games: int = 100):
        """
        Initialize the evaluator.
        
        Args:
            game_class: The class of the game to be played
            num_games: Number of games to play in the evaluator
        """
        self.game_class = game_class
        self.num_games = num_games
        self.results = {}
        self.game_logs = []

    def run_game(self, agent1, agent2, render: bool = False, 
                    training: bool = False, track_metrics: bool = True) -> Dict[str, Any]:
        """
        Run a single game between two agents.

        Args:
            agent1: The first agent (player 1)
            agent2: The second agent (player 2)
            render: Whether to render the game
            training: Whether agents are in training mode
            track_metrics: Whether to track and return metrics
            
        Returns:
            Dictionary with game results and metrics
        """
        game = self.game_class()
        state = game.reset()

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

        while not game.is_terminal():
            current_player = game.get_current_player()

            if current_player == 1:
                # this is player 1's turn
                agent_start_time = time.time()
                action = agent1.select_action(game, training=training)
                metrics["agent1_time"] += time.time() - agent_start_time

                if hasattr(agent1, 'nodes_visited'):
                    metrics["agent1_nodes"] += agent1.nodes_visited

                next_state, reward, done, info = game.step(action)

                if training and hasattr(agent1, 'learn'):
                    agent1.learn(game, reward, done)

            else:
                # this is player 2's turn
                agent_start_time = time.time()
                action = agent2.select_action(game, training=training)
                metrics["agent2_time"] += time.time() - agent_start_time

                if hasattr(agent2, 'nodes_visited'):
                    metrics["agent2_nodes"] += agent2.nodes_visited

                next_state, reward, done, info = game.step(action)

                if training and hasattr(agent2, 'learn'):
                    agent2.learn(game, reward, done)

            if render:
                print(game.render())
                print(f"Player {current_player} chose action: {action}")
                if done:
                    winner = info["winner"]
                    if winner == 0:
                        print("Game ended in a draw.")
                    else:
                        print(f"Player {winner} won!")

            metrics["moves"] += 1

        metrics["winner"] = game.get_winner()
        metrics["game_time"] = time.time() - start_time

        return metrics

    def run_evaluator(self, agents_list: List[Tuple[Any, Any]],
                        symmetric: bool = True,
                        render_final: bool = False,
                        training: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Run a multiple games between multiple agents.

        Args:
            agents_list: List of (agent, name) tuples
            symmetric: Whether to play symmetric games (A vs B and B vs A)
            render_final: Whether to render the final game
            training: Whether agents are in training mode
            
        Returns:
            Dictionary with games results
        """
        results = {}

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

        games = []
        for i, (agent1, name1) in enumerate(agents_list):
            for j, (agent2, name2) in enumerate(agents_list):
                if i == j:  # skip playing against oneself
                    continue
                if not symmetric and j < i:
                    continue
                games.append((agent1, name1, agent2, name2))

        for agent1, name1, agent2, name2 in tqdm(games, desc="Running game evaluator"):
            print(f"\n\nGame: {name1} vs {name2}")

            # clone agents to ensure independence for each game
            agent1_clone = agent1.__class__(agent1.player_id)
            agent2_clone = agent2.__class__(agent2.player_id)

            for attr in ['use_alpha_beta', 'max_depth', 'q_table']:
                if hasattr(agent1, attr):
                    setattr(agent1_clone, attr, getattr(agent1, attr))
                if hasattr(agent2, attr):
                    setattr(agent2_clone, attr, getattr(agent2, attr))

            game_results = []

            # run multiple games
            for game_idx in tqdm(range(self.num_games), desc=f"Games ({name1} vs {name2})"):
                render = render_final and game_idx == self.num_games - 1
                game_result = self.run_game(agent1_clone, agent2_clone,
                                            render=render, training=training)
                game_results.append(game_result)

            wins = sum(1 for g in game_results if g["winner"] == 1)
            losses = sum(1 for g in game_results if g["winner"] == 2)
            draws = sum(1 for g in game_results if g["winner"] == 0)

            avg_moves = sum(g["moves"] for g in game_results) / len(game_results)
            avg_time = sum(g["game_time"] for g in game_results) / len(game_results)

            results[name1]["wins"] += wins
            results[name1]["losses"] += losses
            results[name1]["draws"] += draws
            results[name1]["total_games"] += len(game_results)

            results[name2]["wins"] += losses
            results[name2]["losses"] += wins
            results[name2]["draws"] += draws
            results[name2]["total_games"] += len(game_results)

            games_key = f"{name1} vs {name2}"
            results[games_key] = {
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": wins / len(game_results),
                "draw_rate": draws / len(game_results),
                "avg_moves": avg_moves,
                "avg_time": avg_time,
                "avg_agent1_time": sum(r["agent1_time"]
                                        for r in game_results) / len(game_results),
                "avg_agent2_time": sum(r["agent2_time"]
                                        for r in game_results) / len(game_results),
            }

            if any(r["agent1_nodes"] is not None for r in game_results):
                results[games_key]["avg_agent1_nodes"] = sum(g["agent1_nodes"]
                                                            for g in game_results
                                                            if g["agent1_nodes"] is not None) / len(game_results)

            if any(r["agent2_nodes"] is not None for r in game_results):
                results[games_key]["avg_agent2_nodes"] = sum(g["agent2_nodes"]
                                                            for g in game_results
                                                            if g["agent2_nodes"] is not None) / len(game_results)

            print(f"Results: {wins} wins, {losses} losses, {draws} draws")
            print(f"Win rate: {wins / len(game_results):.2f}")
            print(f"Average moves per game: {avg_moves:.2f}")

        for name in [name for agent, name in agents_list]:
            if results[name]["total_games"] > 0:
                results[name]["win_rate"] = results[name]["wins"] / results[name]["total_games"]
                results[name]["draw_rate"] = results[name]["draws"] / results[name]["total_games"]

        self.results = results
        return results

    def save_results_to_csv(self, path: str) -> None:
        """
        Save evaluator results to a CSV file.

        Args:
            path: Path to save the CSV file
        """
        if not self.results:
            print("No results to save. Run game evaluator first.")
            return

        agent_stats = {}
        for key, value in self.results.items():
            if " vs " not in key:
                agent_stats[key] = value

        df = pd.DataFrame(agent_stats).T
        df.index.name = 'Agent'

        df.to_csv(path)
        print(f"Results saved to {path}")

    def print_detailed_results(self) -> None:
        """Print detailed results of the game evaluator."""
        if not self.results:
            print("No results to print. Run game evaluator first.")
            return

        print("\n===== GAME RESULTS =====\n")

        print("Overall Agent Performance:")
        agent_stats = {}
        for key, value in self.results.items():
            if " vs " not in key:
                agent_stats[key] = value

        df_overall = pd.DataFrame(agent_stats).T
        df_overall['win_rate'] = df_overall['win_rate'].apply(lambda x: f"{x:.2%}")
        df_overall['draw_rate'] = df_overall.get('draw_rate', 0).apply(lambda x: f"{x:.2%}")
        print(df_overall[['wins', 'draws', 'losses', 'win_rate', 'total_games']])

        print("Full game results:")
        for key, value in self.results.items():
            if " vs " in key:
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

    def save_detailed_results_to_csv(self, path: str) -> None:
        """
        Save detailed game results to a CSV file.

        Args:
            path: Path to save the CSV file
        """
        if not self.results:
            print("No results to save. Run game evaluator first.")
            return

        game_stats = {}
        for key, value in self.results.items():
            if " vs " in key:
                game_stats[key] = value

        df = pd.DataFrame(game_stats).T
        df.index.name = 'GameStats'

        detailed_path = path.replace('.csv', '_detailed.csv')
        df.to_csv(detailed_path)
        print(f"Detailed results saved to {detailed_path}")

        return df

    def plot_efficiency_comparison(self, title: str = "Algorithm efficiency comparison",
                                    save_path: Optional[str] = None) -> None:
        """
        Generate separate plots for efficiency metrics.

        Args:
            title: Base title for the plots
            save_path: Base path to save the plots, if provided
        """
        if not self.results:
            print("No results to plot. Run game evaluator first.")
            return

        games = []
        agent1_names = []
        agent2_names = []
        nodes_visited = []
        exec_times = []

        for key, value in self.results.items():
            if " vs " in key and "avg_agent1_nodes" in value:
                parts = key.split(" vs ")
                agent1, agent2 = parts[0], parts[1]

                games.append(key)
                agent1_names.append(agent1)
                agent2_names.append(agent2)
                nodes_visited.append(value["avg_agent1_nodes"])
                exec_times.append(value["avg_agent1_time"])

        if not games:
            print("No efficiency data to plot.")
            return

        # 1. plot for nodes visited
        plt.figure(figsize=(10, 6))
        sorted_indices = np.argsort(nodes_visited)
        print("sorted_indices: ", sorted_indices)
        sorted_games = [games[i] for i in sorted_indices]
        print("sorted_games: ", sorted_games)
        sorted_agent1 = [agent1_names[i] for i in sorted_indices]
        print("sorted_agent1: ", sorted_agent1)
        sorted_nodes = [nodes_visited[i] for i in sorted_indices]
        print("sorted_nodes: ", sorted_nodes)

        labels = []
        for game in sorted_games:
            parts = game.split(" vs ")
            labels.append(f"{parts[0]}")

        unique_labels = tuple(set(labels))
        print("unique_labels: ", unique_labels)
        avg_nodes_alg = []
        for i in range(0, len(sorted_nodes), 3):
            avg_nodes_alg.append(np.mean(sorted_nodes[i:i+3]))

        node_bars = plt.bar(unique_labels, avg_nodes_alg, color='#4285F4')
        plt.title('Nodes visited by each algorithm', fontsize=14)
        plt.ylabel('Average nodes visited', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.3)

        for bar in node_bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            nodes_path = save_path.replace('.png', '_nodes_visited.png')
            plt.savefig(nodes_path, dpi=300, bbox_inches='tight')
            print(f"Nodes visited plot saved to {nodes_path}")

        # 2. plot for execution time
        plt.figure(figsize=(10, 6))
        sorted_times = [exec_times[i] for i in sorted_indices]
        print("sorted_times: ", sorted_times)

        avg_times_alg = []
        for i in range(0, len(sorted_times), 3):
            avg_times_alg.append(np.mean(sorted_times[i:i+3]))

        time_bars = plt.bar(unique_labels, avg_times_alg, color='#34A853')
        plt.title('Execution time by algorithm', fontsize=14)
        plt.ylabel('Average execution time (seconds)', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.3)

        for bar in time_bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.4f}s', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            time_path = save_path.replace('.png', '_execution_time.png')
            plt.savefig(time_path, dpi=300, bbox_inches='tight')
            print(f"Execution time plot saved to {time_path}")

        # 3. plot for Alpha-Beta pruning comparison
        minimax_without = [node for agent, node in zip(
            agent1_names, nodes_visited) if "Minimax without" in agent]
        minimax_with = [node for agent, node in zip(
            agent1_names, nodes_visited) if "Minimax with" in agent and "without" not in agent]

        if minimax_without and minimax_with:
            plt.figure(figsize=(8, 6))

            avg_without = sum(minimax_without) / len(minimax_without)
            avg_with = sum(minimax_with) / len(minimax_with)

            labels = ['Minimax without\nAlpha-Beta', 'Minimax with\nAlpha-Beta']
            values = [avg_without, avg_with]

            bars = plt.bar(labels, values, color=['#EA4335', '#4285F4'], width=0.5)

            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{int(height):,}', ha='center', va='bottom', fontsize=10)

            plt.title('Impact of Alpha-Beta pruning', fontsize=14)
            plt.ylabel('Average nodes visited', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.3)

            # calculate efficiency gain and add annotation
            efficiency = ((avg_without - avg_with) / avg_without) * 100
            plt.annotate(f'{efficiency:.1f}% reduction in nodes',
                    xy=(1, avg_with/3 + (avg_without - avg_with)/5),
                    xytext=(1.3, avg_with + (avg_without - avg_with)/2),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="black", lw=1, alpha=0.9))

            plt.tight_layout()

            if save_path:
                ab_path = save_path.replace('.png', '_alpha_beta_comparison.png')
                plt.savefig(ab_path, dpi=300, bbox_inches='tight')
                print(f"Alpha-Beta comparison saved to {ab_path}")

        # 4. plot for time comparison for Minimax variants
        minimax_without_time = [time for agent, time in zip(
            agent1_names, exec_times) if "Minimax without" in agent]
        minimax_with_time = [time for agent, time in zip(
            agent1_names, exec_times) if "Minimax with" in agent and "without" not in agent]

        if minimax_without_time and minimax_with_time:
            plt.figure(figsize=(8, 6))

            avg_without_time = sum(minimax_without_time) / len(minimax_without_time)
            avg_with_time = sum(minimax_with_time) / len(minimax_with_time)

            labels = ['Minimax without\nAlpha-Beta', 'Minimax with\nAlpha-Beta']
            values = [avg_without_time, avg_with_time]

            bars = plt.bar(labels, values, color=['#EA4335', '#4285F4'], width=0.5)

            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{height:.4f}s', ha='center', va='bottom', fontsize=10)

            plt.title('Execution time comparison with Alpha-Beta pruning effect', fontsize=14)
            plt.ylabel('Average execution time (seconds)', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.3)

            # plot for calculate time savings and add annotation
            time_savings = ((avg_without_time - avg_with_time) / avg_without_time) * 100
            plt.annotate(f'{time_savings:.1f}% reduction in time',
                    xy=(1, avg_with_time/3 + (avg_without_time - avg_with_time)/5),
                    xytext=(1.3, avg_with_time + (avg_without_time - avg_with_time)/2),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="black", lw=1, alpha=0.9))

            plt.tight_layout()

            if save_path:
                time_comp_path = save_path.replace('.png', '_time_comparison.png')
                plt.savefig(time_comp_path, dpi=300, bbox_inches='tight')
                print(f"Time comparison saved to {time_comp_path}")

        plt.show()
