import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D_Baseline(nn.Module):
    """
    A simple 1D CNN for learning Ne history from genotype matrices.
    
    Architecture:
    - Multiple 1D convolution layers to capture local LD patterns
    - Global pooling to aggregate information across all sites
    - Fully connected layers to predict Ne at different time points
    """
    
    def __init__(self, num_haplotypes, max_variants, num_time_points, hidden_dim=128):
        super(CNN1D_Baseline, self).__init__()
        
        self.num_haplotypes = num_haplotypes
        self.max_variants = max_variants
        self.num_time_points = num_time_points
        
        # First, we need to reshape the input from (batch, haplotypes, variants) 
        # to (batch, haplotypes, variants) for 1D convolution along variants
        
        # 1D Convolution layers (conv along the variant dimension)
        self.conv1 = nn.Conv1d(num_haplotypes, 64, kernel_size=10, stride=1, padding=5)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=10, stride=2, padding=5)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=10, stride=2, padding=5)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_time_points)
        
    def forward(self, x):
        # x shape: (batch_size, num_haplotypes, max_variants)
        
        # Apply convolutions
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global pooling: (batch_size, 256, L) -> (batch_size, 256, 1)
        x = self.global_pool(x)
        
        # Flatten: (batch_size, 256, 1) -> (batch_size, 256)
        x = x.squeeze(-1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer - predict log(Ne) at each time point
        x = self.fc3(x)
        
        return x


class CNN1D_Deeper(nn.Module):
    """
    A deeper 1D CNN with residual connections for better performance.
    """
    
    def __init__(self, num_haplotypes, max_variants, num_time_points, hidden_dim=256):
        super(CNN1D_Deeper, self).__init__()
        
        self.num_haplotypes = num_haplotypes
        self.max_variants = max_variants
        self.num_time_points = num_time_points
        
        # Initial convolution
        self.conv_init = nn.Conv1d(num_haplotypes, 64, kernel_size=7, stride=1, padding=3)
        self.bn_init = nn.BatchNorm1d(64)
        
        # Residual blocks
        self.res_block1 = self._make_residual_block(64, 128, stride=2)
        self.res_block2 = self._make_residual_block(128, 256, stride=2)
        self.res_block3 = self._make_residual_block(256, 512, stride=2)
        
        # Global pooling and FC layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(512, hidden_dim)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden_dim, num_time_points)
        
    def _make_residual_block(self, in_channels, out_channels, stride=1):
        """Create a residual block with skip connection."""
        return ResidualBlock1D(in_channels, out_channels, stride)
        
    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn_init(self.conv_init(x)))
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ResidualBlock1D(nn.Module):
    """A simple residual block for 1D convolutions."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        out = F.relu(out)
        return out


class CNN1D_Large(nn.Module):
    """
    A large CNN model designed for big datasets with many time points.
    Features multi-scale convolutions and attention mechanism.
    """
    def __init__(self, num_haplotypes, max_variants, num_time_points, hidden_dim=512):
        super(CNN1D_Large, self).__init__()
        
        self.num_haplotypes = num_haplotypes
        self.max_variants = max_variants
        self.num_time_points = num_time_points
        
        # Multi-scale convolutional layers
        self.conv1 = nn.Conv1d(num_haplotypes, hidden_dim//8, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(hidden_dim//8, hidden_dim//4, kernel_size=11, padding=5, stride=2)
        self.conv3 = nn.Conv1d(hidden_dim//4, hidden_dim//2, kernel_size=7, padding=3, stride=2)
        self.conv4 = nn.Conv1d(hidden_dim//2, hidden_dim, kernel_size=5, padding=2, stride=2)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim//8)
        self.bn2 = nn.BatchNorm1d(hidden_dim//4)
        self.bn3 = nn.BatchNorm1d(hidden_dim//2)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        
        # Residual blocks for deeper learning
        self.res_block1 = ResidualBlock1D(hidden_dim, hidden_dim)
        self.res_block2 = ResidualBlock1D(hidden_dim, hidden_dim)
        
        # Global pooling and dropout
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.4)
        
        # Fully connected layers with more capacity
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, num_time_points)
        
    def forward(self, x):
        # x shape: (batch_size, num_haplotypes, max_variants)
        
        # Multi-scale convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Residual blocks for deeper feature learning
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # Global pooling
        x = self.global_pool(x)  # Shape: (batch_size, hidden_dim, 1)
        x = x.squeeze(-1)  # Shape: (batch_size, hidden_dim)
        
        # Fully connected layers
        x = self.dropout(x)
        x = F.gelu(self.fc1(x))  # Use GELU activation
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x 