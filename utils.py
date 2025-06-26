import torchaudio
import opensmile
import torch
import torch.nn.functional as F

def pad_tensor(tensor, pad_length):
    return F.pad(torch.tensor(tensor), (0, pad_length))

def load_waveform(audio, preprocessor, max_audio_sequence_length, device):
    waveform, sr = torchaudio.load(audio)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
    waveform = waveform.mean(dim=0).unsqueeze(0)  # shape: (1, time)
    waveform = waveform.to(device)
    features, features_length = preprocessor(input_signal=waveform, length=torch.tensor([waveform.shape[1]]).to(device))
    features = features.transpose(1,2)
    features = features.squeeze(0)  # [T, F]
    current_len = features.shape[0]

    if current_len < max_audio_sequence_length:
        pad_len = max_audio_sequence_length - current_len
        features = F.pad(features, (0, 0, 0, pad_len))  # pad rows (T)
    else:
        features = features[:max_audio_sequence_length]

    
    return features, features_length

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.GeMAPSv01b,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

def extract_prosodic_feature(audio, max_audio_sequence_length):
    result = smile.process_file(audio)
    result = result.iloc[:,0:7]
    feature_tensor = torch.tensor(result.values)
    # Pad rows (time steps) to max_audio_sequence_length
    current_len = feature_tensor.shape[0]
    if current_len < max_audio_sequence_length:
        pad_rows = max_audio_sequence_length - current_len
        padded_feature_tensor = F.pad(feature_tensor, (0, 0, 0, pad_rows))  # pad rows
    else:
        padded_feature_tensor = feature_tensor[:max_audio_sequence_length]  # truncate if too long

    # Add batch dimension: [1, T, F]
    return padded_feature_tensor

def custom_audio_collate_fn(batch):
    # batch is a list of tuples, where each tuple is:
    # (input_features, input_lengths, prosodic_features, label)
    input_features_list = [item[0] for item in batch]
    valid_time_frames_list = [item[1] for item in batch]
    prosodic_features_list = [item[2] for item in batch]
    labels_list = [item[3] for item in batch]
    print([i.shape for i in prosodic_features_list])
    collated_input_features = torch.stack(input_features_list, dim=0)
    collated_time_frames = torch.stack(valid_time_frames_list, dim=0)
    collated_prosodic_features = torch.stack(prosodic_features_list, dim=0)
    collated_labels = torch.stack(labels_list, dim=0)

    return collated_input_features, collated_time_frames, collated_prosodic_features, collated_labels