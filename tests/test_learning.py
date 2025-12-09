"""
Learning Effectiveness Testing for 5D Chess AI
Measures how well the MCTS algorithm learns and improves over time
"""
import sys
sys.path.append('..')
from src.super import Chess5D, MCTS, Node
import cupy as cp
import numpy as np
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt

class LearningMetrics:
    def __init__(self):
        self.iteration_results = []
        self.convergence_data = []
        self.policy_stability = []
        self.value_accuracy = []

    def record_iteration(self, iteration, avg_value, policy_entropy, win_rate):
        self.iteration_results.append({
            'iteration': iteration,
            'avg_value': float(avg_value),
            'policy_entropy': float(policy_entropy),
            'win_rate': float(win_rate)
        })

    def calculate_convergence_rate(self):
        if len(self.iteration_results) < 2:
            return 0

        values = [r['avg_value'] for r in self.iteration_results]
        differences = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
        return np.mean(differences)

    def get_learning_curve(self):
        iterations = [r['iteration'] for r in self.iteration_results]
        win_rates = [r['win_rate'] for r in self.iteration_results]
        return iterations, win_rates

class LearningTester:
    def __init__(self, max_time=1, max_turns=25):
        self.max_time = max_time
        self.max_turns = max_turns
        self.metrics = LearningMetrics()

    def test_search_depth_learning(self, base_config, depth_range=(10, 100), step=10):
        """Test how search depth affects learning quality"""
        results = []

        print("\nTesting Search Depth Learning")
        print("=" * 60)

        for num_searches in range(depth_range[0], depth_range[1] + 1, step):
            config = base_config.copy()
            config['num_searches'] = num_searches

            print(f"\nTesting depth: {num_searches} searches")

            game = Chess5D(self.max_time, self.max_turns)
            game_state = game.get_initial_state()
            mcts = MCTS(game, config)

            # Run multiple searches from same position
            start_time = time.time()
            search_results = []

            for _ in range(5):
                mcts_prob_s, mcts_prob_e = mcts.search(game_state.copy())

                # Calculate entropy (measure of uncertainty)
                prob_s_flat = mcts_prob_s.flatten()
                prob_s_flat = prob_s_flat[prob_s_flat > 0]
                if len(prob_s_flat) > 0:
                    entropy = float(-cp.sum(prob_s_flat * cp.log(prob_s_flat + 1e-10)))
                else:
                    entropy = 0

                search_results.append(entropy)

            search_time = time.time() - start_time

            avg_entropy = np.mean(search_results)
            entropy_std = np.std(search_results)

            results.append({
                'num_searches': num_searches,
                'avg_entropy': avg_entropy,
                'entropy_std': entropy_std,
                'search_time': search_time,
                'stability': 1 / (entropy_std + 0.1)  # Lower std = higher stability
            })

            print(f"  Avg Entropy: {avg_entropy:.4f}")
            print(f"  Entropy Std: {entropy_std:.4f}")
            print(f"  Stability: {results[-1]['stability']:.4f}")
            print(f"  Search Time: {search_time:.3f}s")

        return results

    def test_exploration_exploitation_tradeoff(self, base_config, c_values=[0.5, 1.0, 1.41, 2.0, 2.5]):
        """Test different exploration-exploitation parameters"""
        results = []

        print("\nTesting Exploration-Exploitation Tradeoff")
        print("=" * 60)

        for c_value in c_values:
            config = base_config.copy()
            config['C'] = c_value

            print(f"\nTesting C value: {c_value}")

            game = Chess5D(self.max_time, self.max_turns)
            game_state = game.get_initial_state()
            mcts = MCTS(game, config)

            # Perform search and analyze
            start_time = time.time()
            mcts_prob_s, mcts_prob_e = mcts.search(game_state)
            search_time = time.time() - start_time

            # Measure exploration breadth
            nonzero_positions = float(cp.sum(mcts_prob_s > 0))
            max_prob = float(cp.max(mcts_prob_s))

            # Calculate Gini coefficient (measure of inequality/exploitation)
            prob_s_sorted = cp.sort(mcts_prob_s.flatten())
            n = len(prob_s_sorted)
            index = cp.arange(1, n + 1)
            gini = float((2 * cp.sum(index * prob_s_sorted)) / (n * cp.sum(prob_s_sorted)) - (n + 1) / n)

            results.append({
                'c_value': c_value,
                'nonzero_positions': nonzero_positions,
                'max_prob': max_prob,
                'gini_coefficient': gini,
                'search_time': search_time,
                'exploration_score': nonzero_positions * (1 - gini)
            })

            print(f"  Explored Positions: {nonzero_positions:.0f}")
            print(f"  Max Probability: {max_prob:.4f}")
            print(f"  Gini Coefficient: {gini:.4f}")
            print(f"  Exploration Score: {results[-1]['exploration_score']:.2f}")

        return results

    def test_position_evaluation_consistency(self, config, num_trials=10):
        """Test how consistently MCTS evaluates the same position"""
        print("\nTesting Position Evaluation Consistency")
        print("=" * 60)

        game = Chess5D(self.max_time, self.max_turns)
        game_state = game.get_initial_state()
        mcts = MCTS(game, config)

        evaluations = []

        for trial in range(num_trials):
            print(f"Trial {trial + 1}/{num_trials}...", end=' ')

            mcts_prob_s, mcts_prob_e = mcts.search(game_state.copy())

            # Get top move
            top_move_idx = cp.unravel_index(cp.argmax(mcts_prob_s), mcts_prob_s.shape)
            top_move_prob = float(cp.max(mcts_prob_s))

            evaluations.append({
                'trial': trial,
                'top_move': tuple(int(x) for x in top_move_idx),
                'top_move_prob': top_move_prob
            })

            print(f"Top move prob: {top_move_prob:.4f}")

        # Calculate consistency metrics
        top_moves = [e['top_move'] for e in evaluations]
        unique_moves = len(set(top_moves))
        most_common_move = max(set(top_moves), key=top_moves.count)
        consistency_rate = top_moves.count(most_common_move) / len(top_moves)

        probs = [e['top_move_prob'] for e in evaluations]
        prob_mean = np.mean(probs)
        prob_std = np.std(probs)

        results = {
            'unique_moves': unique_moves,
            'consistency_rate': consistency_rate,
            'most_common_move': most_common_move,
            'prob_mean': prob_mean,
            'prob_std': prob_std,
            'coefficient_of_variation': prob_std / prob_mean if prob_mean > 0 else 0
        }

        print(f"\nConsistency Results:")
        print(f"  Unique Moves: {unique_moves}/{num_trials}")
        print(f"  Consistency Rate: {consistency_rate:.2%}")
        print(f"  Probability Mean: {prob_mean:.4f}")
        print(f"  Probability Std: {prob_std:.4f}")
        print(f"  Coefficient of Variation: {results['coefficient_of_variation']:.4f}")

        return results

    def generate_learning_report(self, depth_results, tradeoff_results, consistency_results):
        """Generate comprehensive learning analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"../results/learning_report_{timestamp}.json"

        report = {
            'timestamp': timestamp,
            'search_depth_analysis': depth_results,
            'exploration_exploitation_analysis': tradeoff_results,
            'consistency_analysis': consistency_results,
            'recommendations': self._generate_recommendations(
                depth_results, tradeoff_results, consistency_results
            )
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nLearning report saved to: {report_file}")
        return report

    def _generate_recommendations(self, depth_results, tradeoff_results, consistency_results):
        """Generate optimization recommendations based on test results"""
        recommendations = []

        # Analyze search depth
        if depth_results:
            best_stability = max(depth_results, key=lambda x: x['stability'])
            recommendations.append(
                f"Optimal search depth: {best_stability['num_searches']} searches "
                f"(stability: {best_stability['stability']:.4f})"
            )

        # Analyze exploration-exploitation
        if tradeoff_results:
            best_exploration = max(tradeoff_results, key=lambda x: x['exploration_score'])
            recommendations.append(
                f"Best C value for exploration: {best_exploration['c_value']} "
                f"(score: {best_exploration['exploration_score']:.2f})"
            )

        # Analyze consistency
        if consistency_results:
            if consistency_results['consistency_rate'] < 0.5:
                recommendations.append(
                    "Low consistency detected. Consider increasing search depth or adjusting C parameter."
                )
            if consistency_results['coefficient_of_variation'] > 0.3:
                recommendations.append(
                    "High probability variation. MCTS may benefit from more simulations."
                )

        return recommendations

def main():
    print("=" * 80)
    print("5D Chess AI Learning Effectiveness Testing")
    print("=" * 80)

    tester = LearningTester(max_time=1, max_turns=25)

    base_config = {
        'num_searches': 20,
        'C': 1.41
    }

    # Run learning tests
    depth_results = tester.test_search_depth_learning(base_config, depth_range=(10, 50), step=10)
    tradeoff_results = tester.test_exploration_exploitation_tradeoff(base_config)
    consistency_results = tester.test_position_evaluation_consistency(base_config, num_trials=10)

    # Generate report
    report = tester.generate_learning_report(depth_results, tradeoff_results, consistency_results)

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"{i}. {rec}")
    print("=" * 80)

if __name__ == "__main__":
    main()
