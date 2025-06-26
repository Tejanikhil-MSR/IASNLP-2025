import torchaudio
import opensmile
import torch
import torch.functional as F

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
    padded = pad_tensor(features, max_audio_sequence_length - features.shape[1])
    return features, features_length

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.GeMAPSv01b,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

def extract_prosodic_feature(audio, max_audio_sequence_length):
    result = smile.process_file(audio)
    result = result.iloc[:,0:7]
    feature_tensor = torch.tensor(result.values)
    padded_feature_tensor = pad_tensor(feature_tensor, max_audio_sequence_length - feature_tensor.shape[0])
    return padded_feature_tensor