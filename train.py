import torch.nn as nn
from torch import optim

from model import EncoderProjector, FeatureFusion, AbstractWordLevelFeatureExtractor, StressPredictor


encoder_projector = EncoderProjector(input_feature_dim=80, output_feature_dim=7, target_t=256)
feature_fusion = FeatureFusion(time_frames=256, feature_dim=7)
word_level_extractor = AbstractWordLevelFeatureExtractor(input_temporal_dim=256, input_feature_dim=7, output_feature_dim=7, output_temporal_dim=32)
classifier = StressPredictor(input_temporal_dim=32, input_feature_dim=7)

parameters = list(encoder_projector.parameters()) + list(feature_fusion.parameters()) + list(word_level_extractor.parameters()) + list(classifier.parameters())

optimizer = optim.Adam(parameters, lr=0.001)

criterion = nn.BCEWithLogitsLoss()