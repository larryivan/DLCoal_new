import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism for capturing long-range dependencies."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        
        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        return self.out_proj(attn_output)


class ResidualConvBlock(nn.Module):
    """Advanced residual convolutional block with multiple pathways."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=kernel_size//2 * dilation, 
                              dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                              padding=kernel_size//2, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        residual = self.skip(x)
        
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out += residual
        return F.gelu(out)


class MultiScaleConvBlock(nn.Module):
    """Multi-scale convolution block to capture patterns at different scales."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Different kernel sizes for multi-scale feature extraction
        self.conv_small = nn.Conv1d(in_channels, out_channels//4, kernel_size=3, padding=1)
        self.conv_medium = nn.Conv1d(in_channels, out_channels//4, kernel_size=7, padding=3)
        self.conv_large = nn.Conv1d(in_channels, out_channels//4, kernel_size=15, padding=7)
        self.conv_xlarge = nn.Conv1d(in_channels, out_channels//4, kernel_size=31, padding=15)
        
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        small = self.conv_small(x)
        medium = self.conv_medium(x)
        large = self.conv_large(x)
        xlarge = self.conv_xlarge(x)
        
        # Concatenate multi-scale features
        out = torch.cat([small, medium, large, xlarge], dim=1)
        return F.gelu(self.bn(out))


class MegaDemographicCNN(nn.Module):
    """
    Ultra-deep CNN model for demographic inference from genomic data.
    Designed for maximum performance on large datasets.
    """
    
    def __init__(self, num_haplotypes, max_variants, num_time_points, hidden_dim=1024):
        super().__init__()
        
        self.num_haplotypes = num_haplotypes
        self.max_variants = max_variants
        self.num_time_points = num_time_points
        self.hidden_dim = hidden_dim
        
        # Initial multi-scale feature extraction
        self.initial_conv = MultiScaleConvBlock(num_haplotypes, hidden_dim//4)
        
        # Progressive feature extraction with increasing channels
        self.conv_layers = nn.ModuleList([
            ResidualConvBlock(hidden_dim//4, hidden_dim//2, kernel_size=11, dilation=1),
            ResidualConvBlock(hidden_dim//2, hidden_dim//2, kernel_size=7, dilation=2),
            ResidualConvBlock(hidden_dim//2, hidden_dim, kernel_size=5, dilation=1, stride=2),
            ResidualConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2),
            ResidualConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=1),
            ResidualConvBlock(hidden_dim, hidden_dim*2, kernel_size=3, stride=2),
            ResidualConvBlock(hidden_dim*2, hidden_dim*2, kernel_size=3, dilation=2),
            ResidualConvBlock(hidden_dim*2, hidden_dim*2, kernel_size=3),
        ])
        
        # Self-attention for capturing long-range dependencies
        self.attention = MultiHeadSelfAttention(hidden_dim*2, num_heads=16, dropout=0.2)
        
        # Global pooling strategies
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Feature fusion and prediction layers
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim*4, hidden_dim*2),  # *4 because of avg+max pooling
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        
        # Multi-head prediction for different time scales
        self.short_term_head = nn.Linear(hidden_dim, num_time_points//4)  # Recent times
        self.medium_term_head = nn.Linear(hidden_dim, num_time_points//2)  # Medium times
        self.long_term_head = nn.Linear(hidden_dim, num_time_points//4)   # Ancient times
        
        # Final fusion layer
        self.final_prediction = nn.Linear(num_time_points, num_time_points)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: (batch_size, num_haplotypes, max_variants)
        batch_size = x.size(0)
        
        # Initial multi-scale feature extraction
        x = self.initial_conv(x)
        
        # Progressive convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Self-attention mechanism
        # Reshape for attention: (batch, seq_len, features)
        seq_len = x.size(2)
        x_att = x.transpose(1, 2)  # (batch_size, seq_len, hidden_dim*2)
        x_att = self.attention(x_att)
        x_att = x_att.transpose(1, 2)  # Back to (batch_size, hidden_dim*2, seq_len)
        
        # Residual connection with attention
        x = x + x_att
        
        # Global pooling with multiple strategies
        avg_pooled = self.global_avg_pool(x).squeeze(-1)  # (batch_size, hidden_dim*2)
        max_pooled = self.global_max_pool(x).squeeze(-1)  # (batch_size, hidden_dim*2)
        
        # Combine pooling strategies
        pooled_features = torch.cat([avg_pooled, max_pooled], dim=1)  # (batch_size, hidden_dim*4)
        
        # Feature fusion
        features = self.feature_fusion(pooled_features)  # (batch_size, hidden_dim)
        
        # Multi-scale time prediction
        short_pred = self.short_term_head(features)
        medium_pred = self.medium_term_head(features)
        long_pred = self.long_term_head(features)
        
        # Combine predictions
        combined_pred = torch.cat([short_pred, medium_pred, long_pred], dim=1)
        
        # Final prediction layer
        output = self.final_prediction(combined_pred)
        
        return output
    
    def get_attention_weights(self, x):
        """Get attention weights for visualization."""
        # Forward pass up to attention layer
        x = self.initial_conv(x)
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Get attention weights
        x_att = x.transpose(1, 2)
        attention_weights = self.attention.get_attention_weights(x_att)
        
        return attention_weights


class MegaDemographicCNNLite(nn.Module):
    """
    Lighter version of the mega model for faster training and validation.
    """
    
    def __init__(self, num_haplotypes, max_variants, num_time_points, hidden_dim=512):
        super().__init__()
        
        self.num_haplotypes = num_haplotypes
        self.max_variants = max_variants
        self.num_time_points = num_time_points
        
        # Multi-scale initial convolution
        self.initial_conv = MultiScaleConvBlock(num_haplotypes, hidden_dim//2)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualConvBlock(hidden_dim//2, hidden_dim//2, kernel_size=7),
            ResidualConvBlock(hidden_dim//2, hidden_dim, kernel_size=5, stride=2),
            ResidualConvBlock(hidden_dim, hidden_dim, kernel_size=3),
            ResidualConvBlock(hidden_dim, hidden_dim*2, kernel_size=3, stride=2),
            ResidualConvBlock(hidden_dim*2, hidden_dim*2, kernel_size=3),
        ])
        
        # Attention
        self.attention = MultiHeadSelfAttention(hidden_dim*2, num_heads=8, dropout=0.2)
        
        # Pooling and prediction
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, num_time_points)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Self-attention
        x_att = x.transpose(1, 2)
        x_att = self.attention(x_att)
        x_att = x_att.transpose(1, 2)
        x = x + x_att
        
        # Global pooling and prediction
        x = self.global_pool(x).squeeze(-1)
        output = self.predictor(x)
        
        return output 