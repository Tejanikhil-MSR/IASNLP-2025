import torch
import torch.nn as nn
import torch.nn.functional as F

class StressClassifier(nn.Module):
    def __init__(self, encoder, classifier_head):
        super().__init__()
        self.encoder = encoder
        self.classifier_head = classifier_head

    def forward(self, audio_tensors, valid_frames, prosody_tensor):
        # Step 1: Process raw audio through encoder
        encoder_output,encoder_output_shape = self.encoder(audio_signal=audio_tensors, length=valid_frames)
        encoder_output = encoder_output.transpose(1,2)
        # Step 2: Pass encoder output and prosody features through classifier head
        output = self.classifier_head(encoder_output, prosody_tensor)
        return output

class ClassificationHead(nn.Module):
    def __init__(self, encoder_output_shape, prosody_shape, max_output_seq_length, word_level_feature_dim):
        """
        encoder_output_shape: (T_enc, F_enc) - Temporal & Feature dimensions from encoder
        prosody_shape: (T_pros, F_pros) - Temporal & Feature dimensions of prosody features
        max_output_seq_length: Desired output sequence length (word-level)
        word_level_feature_dim: Dimension for abstract word-level features
        """
        super().__init__()
        T_enc, F_enc = encoder_output_shape
        T_pros, F_pros = prosody_shape

        # Project encoder output to match prosody feature dimensions
        self.projector = EncoderProjector(T_enc, F_enc, T_pros, F_pros)

        # Fuse projected encoder features with prosody features
        self.feature_fusion = FeatureFusion(F_pros)

        # Extract abstract word-level features
        self.ABW_representation = AbstractWordLevelFeatureExtractor(
            T_pros, F_pros, word_level_feature_dim, max_output_seq_length
        )

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(word_level_feature_dim, word_level_feature_dim//2),
            nn.ReLU(),
            nn.Linear(word_level_feature_dim//2, 1)  # Binary stress classification
        )

    def forward(self, encoder_output, prosody_tensor):
        # Project encoder output to match prosody dimensions
        # print("Input of Classification Module : ", encoder_output.shape)
        projected = self.projector(encoder_output)

        # Fuse features
        fused = self.feature_fusion(projected, prosody_tensor)

        # Extract word-level representations
        word_features = self.ABW_representation(fused)

        # Classify each word position
        logits = self.classifier(word_features)
        return logits.squeeze(-1)  # Output shape: (batch, max_output_seq_length)

class EncoderProjector(nn.Module):
    def __init__(self, T_enc, F_enc, T_enc_proj, F_enc_proj):
        super().__init__()
        self.T_enc_proj = T_enc_proj
        self.F_enc = F_enc
        self.T_enc = T_enc
        # Lets first reduce the feature dimension
        self.feature_reducer = nn.Sequential(
            nn.Linear(F_enc, F_enc//2),
            nn.ReLU(),
            nn.Linear(F_enc//2, F_enc // 4),
            nn.ReLU(),
            nn.Linear(F_enc // 4, F_enc_proj)
        )

    def forward(self, x):
        # Reduce feature dimension to (batch, T_enc, F_enc_proj)
        # print(x.shape, self.F_enc, self.T_enc)
        x = self.feature_reducer(x)
        # Adjust temporal dimension of input
        x = x.transpose(1, 2)  # (batch, F_enc_proj, T_enc)

        # Interpolate to target length
        if x.size(2) != self.T_enc_proj:
            # we can use the convolution also where downsampling happens with learnable parameters
            # but to reduce the training cost, we are using interpolation (Deterministic downsampling)
            x = F.interpolate(x, size=self.T_enc_proj, mode='linear', align_corners=False)

        return x.transpose(1, 2)  # (batch, target_t, 7)

class FeatureFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # Learnable weights for feature fusion
        self.fusion_weights = nn.Parameter(torch.ones(feature_dim))
        # Optional: Add small noise for initialization
        nn.init.normal_(self.fusion_weights, mean=1.0, std=0.1)

    def forward(self, projected, prosody):
        # Both inputs: (batch, T, F)
        # Feature-wise weighted fusion
        fused = projected + self.fusion_weights * prosody
        return fused

class AbstractWordLevelFeatureExtractor(nn.Module):
    def __init__(self, T_in, F_in, F_out, T_out):
        super().__init__()
        # Temporal compression
        self.temporal_compressor = nn.Sequential(
            nn.Linear(T_in, T_in//2),
            nn.ReLU(),
            nn.Linear(T_in//2, T_out),
            nn.ReLU()
        )

        # Feature expansion
        self.feature_expander = nn.Sequential(
            nn.Linear(F_in, F_in*2),
            nn.ReLU(),
            nn.Linear(F_in*2, F_in*4),
            nn.ReLU(),
            nn.Linear(F_in*4, F_out)
        )

    def forward(self, x):
        # x shape: (batch, T_in, F_in)
        # Compress temporal dimension
        x = x.permute(0, 2, 1)  # (batch, F_in, T_in)
        x = self.temporal_compressor(x)  # (batch, F_in, T_out)

        # Expand features
        x = x.permute(0, 2, 1)  # (batch, T_out, F_in)
        return self.feature_expander(x)  # (batch, T_out, F_out)
