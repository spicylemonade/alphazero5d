"""
Optimized 5D Chess AI Architecture with Enhanced MCTS and Neural Network Integration
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cupy as cp
import numpy as np
import math
import copy
from super import Chess5D, ChessState, Node, MCTS, Checkmate, Stalemate, DrawLoss


class ResidualBlock(nn.Module):
    """Residual block with batch normalization and skip connections"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class AttentionModule(nn.Module):
    """Multi-head attention for capturing temporal and spatial dependencies"""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        batch, channels, time, turns, height, width = x.shape
        # Reshape for attention: flatten spatial dimensions
        x_flat = x.view(batch, channels, -1).transpose(1, 2)  # (B, T*H*W, C)
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        attn_out = self.norm(attn_out)
        # Reshape back
        return attn_out.transpose(1, 2).view(batch, channels, time, turns, height, width)


class PolicyValueNetwork(nn.Module):
    """
    Enhanced Policy-Value Network for 5D Chess with:
    - Residual connections for deep learning
    - Attention mechanisms for temporal reasoning
    - Separate policy and value heads
    - Multi-scale feature extraction
    """
    def __init__(self, max_time=11, max_turns=30, num_res_blocks=10, channels=256):
        super().__init__()
        self.max_time = max_time
        self.max_turns = max_turns
        self.channels = channels

        # Initial convolution to expand dimensions
        # Input: (batch, max_time, max_turns*2, 6, 8, 8)
        self.conv_initial = nn.Conv3d(6, channels, kernel_size=3, padding=1)
        self.bn_initial = nn.BatchNorm3d(channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_res_blocks)
        ])

        # Attention layers for temporal reasoning
        self.attention1 = AttentionModule(channels, num_heads=8)
        self.attention2 = AttentionModule(channels, num_heads=8)

        # Policy head for move selection (start positions)
        self.policy_conv1 = nn.Conv3d(channels, 64, kernel_size=1)
        self.policy_bn1 = nn.BatchNorm3d(64)
        self.policy_conv2 = nn.Conv3d(64, 32, kernel_size=1)
        self.policy_bn2 = nn.BatchNorm3d(32)

        # Calculate flattened size for policy head
        self.policy_fc_size = 32 * max_time * max_turns * 8 * 8
        self.policy_fc = nn.Linear(self.policy_fc_size, max_time * max_turns * 8 * 8)

        # Policy head for end positions
        self.policy_end_conv1 = nn.Conv3d(channels, 64, kernel_size=1)
        self.policy_end_bn1 = nn.BatchNorm3d(64)
        self.policy_end_conv2 = nn.Conv3d(64, 32, kernel_size=1)
        self.policy_end_bn2 = nn.BatchNorm3d(32)
        self.policy_end_fc = nn.Linear(self.policy_fc_size, max_time * max_turns * 8 * 8)

        # Value head for position evaluation
        self.value_conv = nn.Conv3d(channels, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm3d(32)
        self.value_fc1 = nn.Linear(32 * max_time * max_turns * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, max_time, max_turns*2, 6, 8, 8) - board tensor
        Returns:
            policy_start: (batch, max_time, max_turns, 8, 8) - start position probabilities
            policy_end: (batch, max_time, max_turns, 8, 8) - end position probabilities
            value: (batch, 1) - position evaluation [-1, 1]
        """
        # Reshape to handle 5D input as 3D conv with merged time-turns dimension
        batch_size = x.shape[0]

        # Reshape: (B, T, Turns*2, 6, 8, 8) -> (B, 6, T*Turns*2, 8, 8)
        x = x.permute(0, 3, 1, 2, 4, 5)
        x = x.reshape(batch_size, 6, self.max_time * self.max_turns * 2, 8, 8)

        # Initial processing
        x = F.relu(self.bn_initial(self.conv_initial(x)))

        # Residual tower
        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x)
            # Apply attention at specific layers
            if i == len(self.res_blocks) // 2:
                x = self.attention1(x)
            elif i == len(self.res_blocks) - 1:
                x = self.attention2(x)

        # Policy head (start positions)
        policy_start = F.relu(self.policy_bn1(self.policy_conv1(x)))
        policy_start = F.relu(self.policy_bn2(self.policy_conv2(policy_start)))
        policy_start = policy_start.view(batch_size, -1)
        policy_start = self.policy_fc(policy_start)
        policy_start = policy_start.view(batch_size, self.max_time, self.max_turns, 8, 8)

        # Policy head (end positions)
        policy_end = F.relu(self.policy_end_bn1(self.policy_end_conv1(x)))
        policy_end = F.relu(self.policy_end_bn2(self.policy_end_conv2(policy_end)))
        policy_end = policy_end.view(batch_size, -1)
        policy_end = self.policy_end_fc(policy_end)
        policy_end = policy_end.view(batch_size, self.max_time, self.max_turns, 8, 8)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(batch_size, -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy_start, policy_end, value


class AlphaZeroMCTS(MCTS):
    """
    Enhanced MCTS with neural network guidance (AlphaZero style)
    """
    def __init__(self, game, args, model=None, device='cuda'):
        super().__init__(game, args)
        self.model = model
        self.device = device

    def search(self, state):
        """
        Perform MCTS search with neural network guidance
        """
        root = AlphaZeroNode(self.game, self.args, state, self.model, self.device)

        for search in range(self.args['num_searches']):
            node = root

            # Selection: traverse tree using UCB
            while node.is_fully_expanded():
                node = node.select()

            # Check terminal
            value, is_terminal = node.state.value, node.state.is_terminal
            value = self.game.get_opponent_value(node.state, value)

            if not is_terminal:
                # Expansion: add new child node
                node = node.expand()
                # Evaluation: use neural network instead of rollout
                if self.model is not None:
                    value = node.evaluate_with_network()
                else:
                    # Fallback to simulation
                    value = node.simulate()

            # Backpropagation
            node.backpropagate(value)

        # Return improved action probabilities
        action_probs_start = cp.zeros(self.game.action_size, dtype=cp.float64)
        action_probs_end = cp.zeros(self.game.action_size, dtype=cp.float64)

        for child in root.children:
            action_probs_start[child.action_taken_s] = child.visit_count
            action_probs_end[child.action_taken_e] = child.visit_count

        # Temperature-based selection for exploration
        if 'temperature' in self.args and self.args['temperature'] > 0:
            temp = self.args['temperature']
            action_probs_start = action_probs_start ** (1.0 / temp)
            action_probs_end = action_probs_end ** (1.0 / temp)

        action_probs_start /= cp.sum(action_probs_start)
        action_probs_end /= cp.sum(action_probs_end)

        return action_probs_start, action_probs_end


class AlphaZeroNode(Node):
    """Enhanced Node with neural network evaluation"""
    def __init__(self, game, args, state, model=None, device='cuda', parent=None,
                 action_taken_s=None, action_taken_e=None, prior_prob=1.0):
        super().__init__(game, args, state, parent, action_taken_s, action_taken_e)
        self.model = model
        self.device = device
        self.prior_prob = prior_prob

    def get_ucb(self, child):
        """
        Enhanced UCB with prior probabilities (PUCT algorithm)
        """
        if child.visit_count == 0:
            q_value = 0
        else:
            if self.parent is not None:
                if self.parent.player == child.player:
                    q_value = ((child.value_sum / child.visit_count) + 1) / 2
                else:
                    q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
            else:
                q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2

        # PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        u_value = self.args['C'] * child.prior_prob * math.sqrt(self.visit_count) / (1 + child.visit_count)

        return q_value + u_value

    def evaluate_with_network(self):
        """Evaluate position using neural network"""
        if self.model is None:
            return self.simulate()

        try:
            self.model.eval()
            with torch.no_grad():
                # Convert board to torch tensor
                board_np = cp.asnumpy(self.state.board)
                board_tensor = torch.from_numpy(board_np).float().unsqueeze(0).to(self.device)

                # Get value prediction
                _, _, value = self.model(board_tensor)
                value = value.item()

            return value
        except Exception as e:
            print(f"Network evaluation failed: {e}, falling back to simulation")
            return self.simulate()

    def expand(self):
        """Expand with prior probabilities from network"""
        action, rl_action_s, rl_action_e = self.game.pick_choice(
            self.state, self.expandable_moves_start, self.expandable_moves_end, False
        )

        child_state = self.state.copy()
        self.game.make_move(child_state, action)

        # Get prior probability if model available
        prior_prob = 1.0
        if self.model is not None:
            try:
                self.model.eval()
                with torch.no_grad():
                    board_np = cp.asnumpy(child_state.board)
                    board_tensor = torch.from_numpy(board_np).float().unsqueeze(0).to(self.device)
                    policy_start, policy_end, _ = self.model(board_tensor)

                    # Get probability for this action
                    prior_prob = (
                        policy_start[0, rl_action_s[0], rl_action_s[1], rl_action_s[2], rl_action_s[3]].item() +
                        policy_end[0, rl_action_e[0], rl_action_e[1], rl_action_e[2], rl_action_e[3]].item()
                    ) / 2.0
            except Exception as e:
                print(f"Prior probability extraction failed: {e}")
                prior_prob = 1.0

        child = AlphaZeroNode(
            self.game, self.args, child_state, self.model, self.device,
            self, rl_action_s, rl_action_e, prior_prob
        )

        self.children.append(child)
        return child


class AdaptiveMCTS:
    """
    Adaptive MCTS that adjusts search depth based on position complexity
    """
    def __init__(self, game, base_args, model=None, device='cuda'):
        self.game = game
        self.base_args = base_args
        self.model = model
        self.device = device

    def calculate_complexity(self, state):
        """Estimate position complexity based on available moves and board state"""
        num_moves = len(state.moves) if state.moves else 0
        board_pieces = cp.count_nonzero(state.board)

        # Normalize complexity score
        move_complexity = min(num_moves / 50.0, 1.0)
        piece_complexity = board_pieces / 32.0

        return (move_complexity + piece_complexity) / 2.0

    def search(self, state):
        """Perform adaptive MCTS with complexity-based search budget"""
        complexity = self.calculate_complexity(state)

        # Adjust search count based on complexity
        min_searches = self.base_args.get('min_searches', 10)
        max_searches = self.base_args.get('num_searches', 100)
        adaptive_searches = int(min_searches + (max_searches - min_searches) * complexity)

        args = self.base_args.copy()
        args['num_searches'] = adaptive_searches

        # Use AlphaZero MCTS with adaptive budget
        mcts = AlphaZeroMCTS(self.game, args, self.model, self.device)
        return mcts.search(state)


def create_training_sample(state, mcts_policy_start, mcts_policy_end, outcome):
    """
    Create training sample for neural network

    Args:
        state: Game state
        mcts_policy_start: MCTS-improved policy for start positions
        mcts_policy_end: MCTS-improved policy for end positions
        outcome: Game outcome from perspective of current player

    Returns:
        Dictionary with training data
    """
    return {
        'board': cp.asnumpy(state.board),
        'policy_start': cp.asnumpy(mcts_policy_start),
        'policy_end': cp.asnumpy(mcts_policy_end),
        'value': outcome,
        'player': state.player
    }
