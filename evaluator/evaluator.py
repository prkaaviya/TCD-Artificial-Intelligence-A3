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

    def plot_efficiency_comparison(self,
                                    title: str = "Algorithm Efficiency Comparison",
                                    save_path: Optional[str] = None) -> None:
        """
        Plot an efficiency comparison focusing on nodes visited and execution time.

        Args:
            title: Title for the plot
            save_path: Path to save the plot, if provided
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

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))

        sorted_indices = np.argsort(nodes_visited)
        sorted_games = [games[i] for i in sorted_indices]
        sorted_agent1 = [agent1_names[i] for i in sorted_indices]
        sorted_nodes = [nodes_visited[i] for i in sorted_indices]
        sorted_times = [exec_times[i] for i in sorted_indices]

        node_bars = ax1.bar(sorted_games, sorted_nodes, color='skyblue')
        ax1.set_title('Nodes visited per algorithm', fontsize=16)
        ax1.set_ylabel('Average nodes visited', fontsize=14)
        ax1.tick_params(axis='x', rotation=45, labelsize=12)

        for bar in node_bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=10)

        time_bars = ax2.bar(sorted_games, sorted_times, color='lightgreen')
        ax2.set_title('Execution time per algorithm', fontsize=16)
        ax2.set_ylabel('Average execution time (seconds)', fontsize=14)
        ax2.tick_params(axis='x', rotation=45, labelsize=12)

        for bar in time_bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.4f}s', ha='center', va='bottom', fontsize=10)

        # for Minimax variants, let's do a special comparison
        minimax_without = [node for agent, node in zip(
            agent1_names, nodes_visited) if "Minimax without" in agent]
        minimax_with = [node for agent, node in zip(
            agent1_names, nodes_visited) if "Minimax with" in agent and "without" not in agent]

        if minimax_without and minimax_with:
            plt.figure(figsize=(10, 6))

            avg_without = sum(minimax_without) / len(minimax_without)
            avg_with = sum(minimax_with) / len(minimax_with)

            labels = ['Minimax without\nAlpha-Beta', 'Minimax with\nAlpha-Beta']
            values = [avg_without, avg_with]

            bars = plt.bar(labels, values, color=['#ff9999', '#66b3ff'])

            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{int(height):,}', ha='center', va='bottom', fontsize=12)

            plt.title('Impact of Alpha-Beta pruning on nodes visited', fontsize=16)
            plt.ylabel('Average nodes visited', fontsize=14)

            efficiency = ((avg_without - avg_with) / avg_without) * 100
            plt.figtext(0.5, 0.01,
                        f'Alpha-Beta pruning reduces nodes visited by {efficiency:.1f}%',
                        ha='center', fontsize=14, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

            plt.tight_layout(rect=[0, 0.05, 1, 0.95])

            if save_path:
                ab_path = save_path.replace('.png', '_alpha_beta_comparison.png')
                plt.savefig(ab_path, dpi=300, bbox_inches='tight')
                print(f"Alpha-Beta comparison saved to {ab_path}")

        plt.figure(fig.number)
        plt.tight_layout()

        if save_path:
            efficiency_path = save_path.replace('.png', '_efficiency.png')
            plt.savefig(efficiency_path, dpi=300, bbox_inches='tight')
            print(f"Efficiency comparison saved to {efficiency_path}")

        plt.show()
