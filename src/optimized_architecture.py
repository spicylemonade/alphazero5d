"""
Optimized Neural Network Architecture for 5D Chess Learning
Features:
- Improved residual connections with attention mechanisms
- Multi-head policy networks for start/end move prediction
- Advanced value estimation with uncertainty quantification
- Gradient flow optimization
- Memory-efficient implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for focusing on important board positions"""
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(channels, 1, kernel_size=1)

    def forward(self, x):
        attention = torch.sigmoid(self.conv(x))
        return x * attention


class ImprovedResidualBlock(nn.Module):
    """Enhanced residual block with attention and normalization"""
    def __init__(self, channels, use_attention=True):
        super(ImprovedResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)
        self.attention = SpatialAttention(channels) if use_attention else None
        self.dropout = nn.Dropout3d(0.1)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        if self.attention is not None:
            out = self.attention(out)

        out += residual
        out = F.relu(out)
        return out


class PolicyHead(nn.Module):
    """Advanced policy head with better move prediction"""
    def __init__(self, in_channels, board_size, output_size):
        super(PolicyHead, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 32, kernel_size=1)
        self.bn2 = nn.BatchNorm3d(32)

        # Calculate flattened size dynamically
        self.flatten_size = 32 * board_size[0] * board_size[1] * board_size[2]

        self.fc1 = nn.Linear(self.flatten_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ValueHead(nn.Module):
    """Enhanced value head with uncertainty estimation"""
    def __init__(self, in_channels, board_size):
        super(ValueHead, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 16, kernel_size=1)
        self.bn2 = nn.BatchNorm3d(16)

        self.flatten_size = 16 * board_size[0] * board_size[1] * board_size[2]

        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)  # Value output
        self.fc_uncertainty = nn.Linear(128, 1)  # Uncertainty estimate
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        value = torch.tanh(self.fc3(x))
        uncertainty = torch.sigmoid(self.fc_uncertainty(x))

        return value, uncertainty


class OptimizedChess5DNet(nn.Module):
    """
    Optimized neural network for 5D Chess with improved architecture:
    - Better gradient flow through residual connections
    - Attention mechanisms for focusing on critical positions
    - Separate optimized heads for policy and value
    - Uncertainty quantification for better decision making
    """
    def __init__(self, input_shape=(6, 11, 60, 8, 8), num_residual_blocks=10,
                 use_attention=True, action_size=(11, 30, 8, 8)):
        super(OptimizedChess5DNet, self).__init__()

        self.input_shape = input_shape
        self.action_size = action_size

        # Input processing
        self.input_conv = nn.Conv3d(input_shape[0], 128, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm3d(128)

        # Residual backbone
        self.residual_blocks = nn.ModuleList([
            ImprovedResidualBlock(128, use_attention=use_attention)
            for _ in range(num_residual_blocks)
        ])

        # Calculate board size for heads
        board_size = (input_shape[1], input_shape[2], input_shape[3] * input_shape[4])
        output_size = action_size[0] * action_size[1] * action_size[2] * action_size[3]

        # Policy heads
        self.policy_start_head = PolicyHead(128, board_size, output_size)
        self.policy_end_head = PolicyHead(128, board_size, output_size)

        # Value head
        self.value_head = ValueHead(128, board_size)

    def forward(self, x):
        # Input processing
        x = F.relu(self.input_bn(self.input_conv(x)))

        # Residual tower
        for block in self.residual_blocks:
            x = block(x)

        # Generate outputs
        policy_start = self.policy_start_head(x)
        policy_end = self.policy_end_head(x)
        value, uncertainty = self.value_head(x)

        # Reshape policy outputs
        policy_start = policy_start.view(-1, *self.action_size)
        policy_end = policy_end.view(-1, *self.action_size)

        return policy_start, policy_end, value, uncertainty


class LightweightChess5DNet(nn.Module):
    """Lightweight version for faster training and testing"""
    def __init__(self, input_shape=(6, 11, 60, 8, 8), num_residual_blocks=5,
                 action_size=(11, 30, 8, 8)):
        super(LightweightChess5DNet, self).__init__()

        self.input_shape = input_shape
        self.action_size = action_size

        # Smaller input processing
        self.input_conv = nn.Conv3d(input_shape[0], 64, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm3d(64)

        # Fewer residual blocks
        self.residual_blocks = nn.ModuleList([
            ImprovedResidualBlock(64, use_attention=False)
            for _ in range(num_residual_blocks)
        ])

        board_size = (input_shape[1], input_shape[2], input_shape[3] * input_shape[4])
        output_size = action_size[0] * action_size[1] * action_size[2] * action_size[3]

        # Simplified heads
        self.policy_start_head = PolicyHead(64, board_size, output_size)
        self.policy_end_head = PolicyHead(64, board_size, output_size)
        self.value_head = ValueHead(64, board_size)

    def forward(self, x):
        x = F.relu(self.input_bn(self.input_conv(x)))

        for block in self.residual_blocks:
            x = block(x)

        policy_start = self.policy_start_head(x)
        policy_end = self.policy_end_head(x)
        value, uncertainty = self.value_head(x)

        policy_start = policy_start.view(-1, *self.action_size)
        policy_end = policy_end.view(-1, *self.action_size)

        return policy_start, policy_end, value, uncertainty


def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the architectures
    print("Testing Optimized Architecture:")
    model_opt = OptimizedChess5DNet()
    print(f"Parameters: {count_parameters(model_opt):,}")

    # Test forward pass
    dummy_input = torch.randn(1, 6, 11, 60, 8, 8)
    ps, pe, v, u = model_opt(dummy_input)
    print(f"Policy Start shape: {ps.shape}")
    print(f"Policy End shape: {pe.shape}")
    print(f"Value shape: {v.shape}")
    print(f"Uncertainty shape: {u.shape}")

    print("\nTesting Lightweight Architecture:")
    model_light = LightweightChess5DNet()
    print(f"Parameters: {count_parameters(model_light):,}")
    ps, pe, v, u = model_light(dummy_input)
    print(f"Policy Start shape: {ps.shape}")
    print(f"Policy End shape: {pe.shape}")
