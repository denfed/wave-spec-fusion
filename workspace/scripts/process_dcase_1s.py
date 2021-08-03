import sys
sys.path.append("../")
import librosa
import torch
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import pathlib

train = pd.read_csv("data/tau_audiovisual_2021/evaluation_setup/fold1_train.csv", sep='\t')
val = pd.read_csv("data/tau_audiovisual_2021/evaluation_setup/fold1_evaluate.csv", sep='\t')


new_df = pd.DataFrame(columns=["filename_wave", "filename_spec", "scene_label"])

audio_dir = "data/tau_audiovisual_2021/"
spec_cachedir = "data/tau_audiovisual_2021/spectrogram_2048window_256hop_cache"
wave_cachedir = "data/tau_audiovisual_2021/waveform_cache"

for idx, row in tqdm(train.iterrows(), total=len(train)):
    audio, sr = librosa.load(os.path.join(audio_dir, row['filename_audio']), sr=48000)
    padded = np.zeros(480000, dtype='float32')
    wave = audio[:480000]
    padded[0:len(wave)] = wave
    
    
    # WAVEFORM
    wave = padded
    
    for idx, i in enumerate(np.split(wave, 10)):
        spec = librosa.feature.melspectrogram(i, n_fft=2048, hop_length=256, n_mels=128, sr=48000, fmin=0, fmax=24000)
#         print(spec.shape)
#         print(spec)
#         if spec.shape[1] == 501:
#             spec = spec[:,:-1]
        spec = np.log(spec)
#         print(spec)

        # z-score normalization
        std = spec.std()
        mean = spec.mean()
        spec = (spec - mean) / std
        
#         print(row['filename_audio'])
        s = row['filename_audio'].replace(".wav", "")
    
        fname_wave = os.path.join(wave_cachedir, f"{s}_{idx}.npy")
        fname_spec = os.path.join(spec_cachedir, f"{s}_{idx}.npy")

        # set paths for wave and spec
        pathlib.Path(os.path.join(spec_cachedir, f"{s}_{idx}.npy")).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(wave_cachedir, f"{s}_{idx}.npy")).parent.mkdir(parents=True, exist_ok=True)
        
        # np.save(os.path.join(self.spectrogram_cachedir, item['filename_audio']).replace(".wav", ".npy"), spec)
        np.save(os.path.join(spec_cachedir, f"{s}_{idx}.npy"), spec)
        np.save(os.path.join(wave_cachedir, f"{s}_{idx}.npy"), i)
        
        new_df = new_df.append({"filename_wave": fname_wave, "filename_spec": fname_spec, "scene_label": row['scene_label']}, ignore_index=True)
    
    
new_df.to_csv("data/tau_audiovisual_2021/train_1sec.csv")


### VALIDATION


new_df = pd.DataFrame(columns=["filename_wave", "filename_spec", "scene_label"])

audio_dir = "data/tau_audiovisual_2021/"
spec_cachedir = "data/tau_audiovisual_2021/spectrogram_2048window_256hop_cache"
wave_cachedir = "data/tau_audiovisual_2021/waveform_cache"

for idx, row in tqdm(val.iterrows(), total=len(val)):
    audio, sr = librosa.load(os.path.join(audio_dir, row['filename_audio']), sr=48000)
    padded = np.zeros(480000, dtype='float32')
    wave = audio[:480000]
    padded[0:len(wave)] = wave
    
    
    # WAVEFORM
    wave = padded
    
    for idx, i in enumerate(np.split(wave, 10)):
        spec = librosa.feature.melspectrogram(i, n_fft=2048, hop_length=256, n_mels=128, sr=48000, fmin=0, fmax=24000)
#         print(spec.shape)
#         print(spec)
#         if spec.shape[1] == 501:
#             spec = spec[:,:-1]

        spec = np.log(spec)
#         print(spec)

        # z-score normalization
        std = spec.std()
        mean = spec.mean()
        spec = (spec - mean) / std
        
#         print(row['filename_audio'])
        s = row['filename_audio'].replace(".wav", "")
    
        fname_wave = os.path.join(wave_cachedir, f"{s}_{idx}.npy")
        fname_spec = os.path.join(spec_cachedir, f"{s}_{idx}.npy")

        # set paths for wave and spec
        pathlib.Path(os.path.join(spec_cachedir, f"{s}_{idx}.npy")).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(wave_cachedir, f"{s}_{idx}.npy")).parent.mkdir(parents=True, exist_ok=True)
        
        # np.save(os.path.join(self.spectrogram_cachedir, item['filename_audio']).replace(".wav", ".npy"), spec)
        np.save(os.path.join(spec_cachedir, f"{s}_{idx}.npy"), spec)
        np.save(os.path.join(wave_cachedir, f"{s}_{idx}.npy"), i)
        
        new_df = new_df.append({"filename_wave": fname_wave, "filename_spec": fname_spec, "scene_label": row['scene_label']}, ignore_index=True)
    
    
new_df.to_csv("data/tau_audiovisual_2021/val_1sec.csv")

print("Finished processing dataset to 1-second samples!")