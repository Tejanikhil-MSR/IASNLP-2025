import torch
import torch.functional as F
import pandas as pd
import torchaudio
import opensmile
import os

class AudioDataset(torch.utils.data.Dataset):
  def __init__(self, audio_paths, csv_path, preprocessor, device, max_audio_sequence_length=1325, max_token_seq_length=32):
    self.audio_paths = audio_paths
    self.max_audio_sequence_length = max_audio_sequence_length
    self.max_output_length = max_token_seq_length
    self.input = []
    self.device=device
    self.prosodic_features = []
    self.label = []
    self.preprocessor = preprocessor

    self.smile = opensmile.Smile(
      feature_set=opensmile.FeatureSet.GeMAPSv01b,
      feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )

    df = pd.read_excel(csv_path)
    df = df.drop("Transcript", axis=1)
    merged_df = df.copy()
    merged_df['Label'] = df.iloc[:, 1:].values.tolist()
    merged_df['Label'] = merged_df['Label'].apply(lambda x: [i for i in x if pd.notna(i)])
    # Keep only 'Audio Link' and the new merged column
    res = merged_df[['Audio Link', 'Label']].to_dict(orient="records")
    for i in res:
      if "denoised_"+i["Audio Link"]+".wav" in os.listdir(audio_paths):
        self.input.append(self.phase1_preprocessing(audio_paths + "/denoised_" + i["Audio Link"] + ".wav"))
        self.label.append(self.pad_tensor(i["Label"], self.max_output_length - len(i["Label"])))
        prosodic_feature = self.extract_prosodic_feature(audio_paths + "/denoised_" + i["Audio Link"] + ".wav")
        self.prosodic_features.append(prosodic_feature)

  def __getitem__(self, index):
    return self.input[index], self.prosodic_features[index], self.label[index]

  def extract_prosodic_feature(self, audio):
    result = self.smile.process_file(audio)
    result = result.iloc[:,0:7]
    feature_tensor = torch.tensor(result.values)
    padded_feature_tensor = self.pad_tensor(feature_tensor, self.max_audio_sequence_length - feature_tensor.shape[0])
    return padded_feature_tensor

  def load_waveform(self, audio):
    waveform, sr = torchaudio.load(audio)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
    waveform = waveform.mean(dim=0).unsqueeze(0)  # shape: (1, time)
    waveform = waveform.to(self.device)
    features, features_length = self.preprocessor(input_signal=waveform, length=torch.tensor([waveform.shape[1]]).to(self.device))
    return features, features_length

  def pad_tensor(self, tensor, pad_length):
    return F.pad(torch.tensor(tensor), (0, pad_length))

  def phase1_preprocessing(self, audio):
    features, features_length = self.load_waveform(audio)
    batch_size, feature_size, valid_time_frames = features.shape
    padded = self.pad_tensor(features, self.max_audio_sequence_length - valid_time_frames)
    return tuple(padded, valid_time_frames)

  def __len__(self):
      return len(self.input)