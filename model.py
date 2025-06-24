import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderProjector(nn.Module):
    def __init__(self, input_feature_dim, output_feature_dim=7, target_t=256):
        super().__init__()
        self.output_temporal_dim = target_t
        self.feature_reducer = nn.Sequential(
            nn.Linear(input_feature_dim, input_feature_dim//2),
            nn.ReLU(),
            nn.Linear(input_feature_dim//2, input_feature_dim // 4),
            nn.ReLU(),
            nn.Linear(input_feature_dim // 4, output_feature_dim)
        )
        
    def forward(self, x):
        # Reduce feature dimension to (batch, T, 7)
        x = self.feature_reducer(x) 
        
        # Adjust temporal dimension
        x = x.transpose(1, 2)  # (batch, 7, T)
        
        # Interpolate to target length
        if x.size(2) != self.output_temporal_dim:
            # we can use the convolution also where downsampling happens with learnable parameters 
            # but to reduce the training cost, we are using interpolation (Deterministic downsampling)
            x = F.interpolate(x, size=self.output_temporal_dim, mode='linear', align_corners=False)
        
        return x.transpose(1, 2)  # (batch, target_t, 7)

class FeatureFusion(nn.Module):
    def __init__(self, time_frames=256, feature_dim=7):
        super().__init__()
        self.time_frames = time_frames
        self.feature_dim = feature_dim
        
        # Fusion layer: learnable weights for each feature
        self.fusion_weights = nn.Parameter(torch.ones(feature_dim))
    
    def forward(self, feature_matrix1, prosodic_fm):
        # Ensure feature_matrix1 and prosodic_fm have compatible dimensions
        if feature_matrix1.size(1) != prosodic_fm.size(1):
            raise ValueError("Feature matrix and prosodic features must have the same temporal dimension.")
        
        # Fuse features: element-wise multiplication with learnable weights
        fused_features = feature_matrix1 + self.fusion_weights * prosodic_fm
        
        return fused_features

class AbstractWordLevelFeatureExtractor(nn.Module):
    def __init__(self, input_temporal_dim = 256, input_feature_dim = 7, output_feature_dim = 7, output_temporal_dim=32):
        super().__init__()
        # I cannot use the convolution here, so lets use linear layers for temporal downsampling
        # Apply linear layers along temporal dimension
        self.temporal_downsampler = nn.Sequential(
            nn.Linear(input_temporal_dim, input_temporal_dim//2),
            nn.ReLU(),
            nn.Linear(input_temporal_dim//2, output_temporal_dim),
            nn.ReLU()
        )
        kernel_size = input_feature_dim + 1 - output_feature_dim
        self.conv = nn.Conv1d(in_channels=output_temporal_dim, 
                              out_channels=output_temporal_dim, 
                              kernel_size=kernel_size,
                              stride=1,
                             )
        
    def forward(self, x):
        # Apply temporal downsampling
        x = x.transpose(1, 2)  # (batch, input_feature_dim, input_temporal_dim)
        x = self.temporal_downsampler(x)  # (batch, input_feature_dim, output_temporal_dim)
        # Apply convolution for feature extraction
        x = x.transpose(1, 2)  # (batch, output_temporal_dim, input_feature_dim)
        x = self.conv(x)  # (batch, output_temporal_dim, output_feature_dim)

        return x  # (batch, output_temporal_dim, output_feature_dim)

# Full stress prediction model
class StressClassifier(nn.Module):
    def __init__(self, input_temporal_dim = 32, input_feature_dim=7, hidden_dim=128):
        super().__init__()
        self.classification_layer = nn.Linear(input_feature_dim, input_temporal_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(input_temporal_dim, 1)  

    def forward(self, x):
        x = self.classification_layer(x)  # (batch, output_temporal_dim, 1)
        x = self.relu(x)  # Apply ReLU activation
        x = self.output_layer(x)  # (batch, output_temporal_dim, 1)
        return x