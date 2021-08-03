from torchvision import datasets
import torchvision.transforms as tv_transforms
from base import BaseDataLoader
import torch.utils.data as data
import librosa
import os
import pandas as pd
import torch
import numpy as np
import pathlib
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, TimeMask
from utils.spec_timeshift_transform import TimeShift

def load_dataframe(path):
    df = pd.read_csv(path)
    return df.to_dict('records')

def load_numpy(path):
    try:
        data = np.load(path)
    except Exception as e:
        print(e)
        return None
    return data


class SpectrogramAugmentationDataset(data.Dataset):
    """
    
    """
    
    def __init__(self, data_dir, label_list, cache_dir, t_shift=False):
        self.data_arr = load_dataframe(data_dir)
        self.data_dir = data_dir
        self.label_list = label_list
        self.spectrogram_cachedir = cache_dir
        
        transforms = []
        if t_shift:
            transforms.append(TimeShift())
            
        if len(transforms)==0:
            self.transform = None
        else:
            self.transform = tv_transforms.Compose(transforms)
        
    def __len__(self):
        return len(self.data_arr)
    
    def __getitem__(self, idx):
        item = self.data_arr[idx]
        scene_label = item['scene_label']
        scene_encoded = self.label_list.index(scene_label)
        
        spec = load_numpy(item['filename_spec'])
        
#         if os.path.isfile(os.path.join(self.spectrogram_cachedir, item['filename_audio']).replace(".wav", ".npy")):
#             spec = load_numpy(os.path.join(self.spectrogram_cachedir, item['filename_audio']).replace(".wav", ".npy"))
#         else:
#             y, sr = librosa.load(os.path.join("/mnt/ssd/data/tau_audiovisual_2021/", item['filename_audio']), sr=48000)
#             spec = librosa.feature.melspectrogram(y, n_fft=2048, hop_length=960, n_mels=256, sr=48000, fmin=0, fmax=22050)
#             if spec.shape[1] == 501:
#                 spec = spec[:,:-1]
            
#             spec = np.log(spec)
            
#             # z-score normalization
#             std = spec.std()
#             mean = spec.mean()
#             spec = (spec - mean) / std
            
#             pathlib.Path(os.path.join(self.spectrogram_cachedir, item['filename_audio']).replace(".wav", ".npy")).parent.mkdir(parents=True, exist_ok=True)
            
#             np.save(os.path.join(self.spectrogram_cachedir, item['filename_audio']).replace(".wav", ".npy"), spec)
        
        if self.transform is not None:
            spec = self.transform(spec)
        
#         return spec, scene_encoded
        return {"spec": spec}, scene_encoded


class WaveformAugmentationDataset(data.Dataset):
    """
    
    """
    
    def __init__(self, data_dir, label_list, cache_dir, 
                 t_gaussian_noise=False,
                 t_time_stretch=False,
                 t_pitch_shift=False,
                 t_shift=False,
                 t_time_mask=False):
        self.data_arr = load_dataframe(data_dir)
        self.data_dir = data_dir
        self.label_list = label_list
        self.waveform_cachedir = cache_dir
        
        transforms = []
        if t_gaussian_noise:
            transforms.append(AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0))
        if t_time_stretch:
            transforms.append(TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0))
        if t_pitch_shift:
            transforms.append(PitchShift(min_semitones=-4, max_semitones=4, p=1.0))
        if t_shift:
            transforms.append(Shift(min_fraction=-0.5, max_fraction=0.5, p=1.0))
        if t_time_mask:
            transforms.append(TimeMask(min_band_part=0.0, max_band_part=0.5, p=1.0))
        
        if len(transforms)==0:
            self.transform = None
        else:
            self.transform = Compose(transforms)
        
    def __len__(self):
        return len(self.data_arr)
    
    def __getitem__(self, idx):
        item = self.data_arr[idx]
        scene_label = item['scene_label']
        scene_encoded = self.label_list.index(scene_label)
        
        wave = load_numpy(item['filename_wave'])

#         if os.path.isfile(os.path.join(self.waveform_cachedir, item['filename_audio']).replace(".wav", ".npy")):
#             wave = load_numpy(os.path.join(self.waveform_cachedir, item['filename_audio']).replace(".wav", ".npy"))
#         else:
#             y, sr = librosa.load(os.path.join("/mnt/ssd/data/tau_audiovisual_2021/", item['filename_audio']), sr=48000)
            
#             padded = np.zeros(480000, dtype='float32')
#             wave = y[:480000]
#             padded[0:len(wave)] = wave
#             wave = padded
            
#             pathlib.Path(os.path.join(self.waveform_cachedir, item['filename_audio']).replace(".wav", ".npy")).parent.mkdir(parents=True, exist_ok=True)

#             np.save(os.path.join(self.waveform_cachedir, item['filename_audio']).replace(".wav", ".npy"), wave)
        
        if self.transform is not None:
#             wave = np.expand_dims(wave, axis=0)
            wave = self.transform(wave, sample_rate=48000)
#             wave = wave.squeeze(0) 
        
        return {"wave": wave}, scene_encoded
    
    

class MultiModalAugmentationDataset(data.Dataset):
    """
    
    """
    
    def __init__(self, data_dir, label_list, spec_cache_dir, wave_cache_dir, w_shift=False, s_shift=False):
        self.data_arr = load_dataframe(data_dir)
        self.data_dir = data_dir
        self.label_list = label_list
        self.w_shift = w_shift
        self.s_shift = s_shift
        self.spectrogram_cachedir = spec_cache_dir
        self.waveform_cachedir = wave_cache_dir
        
        
        spec_transforms = []
        if s_shift:
            spec_transforms.append(TimeShift())
            
        if len(spec_transforms)==0:
            self.spec_transform = None
        else:
            self.spec_transform = tv_transforms.Compose(spec_transforms)
            
        wave_transforms = []
        if w_shift:
            wave_transforms.append(Shift(min_fraction=-0.5, max_fraction=0.5, p=1.0))
        
        if len(wave_transforms)==0:
            self.wave_transform = None
        else:
            self.wave_transform = Compose(wave_transforms)
        
    def __len__(self):
        return len(self.data_arr)
    
    def _get_spec(self, filename):
#         if os.path.isfile(os.path.join(self.spectrogram_cachedir, filename).replace(".wav", ".npy")):
#             spec = load_numpy(os.path.join(self.spectrogram_cachedir, filename).replace(".wav", ".npy"))
#         else:
#             y, sr = librosa.load(os.path.join("/mnt/ssd/data/tau_audiovisual_2021/", filename), sr=48000)
#             spec = librosa.feature.melspectrogram(y, n_fft=2048, hop_length=960, n_mels=40, sr=48000, fmin=0, fmax=22050)
#             if spec.shape[1] == 501:
#                 spec = spec[:,:-1]
            
#             spec = np.log(spec)
            
#             pathlib.Path(os.path.join(self.spectrogram_cachedir, filename).replace(".wav", ".npy")).parent.mkdir(parents=True, exist_ok=True)
            
#             np.save(os.path.join(self.spectrogram_cachedir, filename).replace(".wav", ".npy"), spec)
        return load_numpy(filename)
    
    def _get_wave(self, filename):
#         if os.path.isfile(os.path.join(self.waveform_cachedir, filename).replace(".wav", ".npy")):
#             wave = load_numpy(os.path.join(self.waveform_cachedir, filename).replace(".wav", ".npy"))
#         else:
#             y, sr = librosa.load(os.path.join("/mnt/ssd/data/tau_audiovisual_2021/", filename), sr=48000)
            
#             padded = np.zeros(480000, dtype='float32')
#             wave = y[:480000]
#             padded[0:len(wave)] = wave
#             wave = padded
            
#             pathlib.Path(os.path.join(self.waveform_cachedir, filename).replace(".wav", ".npy")).parent.mkdir(parents=True, exist_ok=True)

#             np.save(os.path.join(self.waveform_cachedir, filename).replace(".wav", ".npy"), wave)
        return load_numpy(filename)
    
    def __getitem__(self, idx):
        item = self.data_arr[idx]
        scene_label = item['scene_label']
        scene_encoded = self.label_list.index(scene_label)
            
        spec = self._get_spec(item['filename_spec'])
        wave = self._get_wave(item['filename_wave'])
        
        # Add transforms
        if self.wave_transform is not None:
            wave = self.wave_transform(wave, sample_rate=48000)
            
        if self.spec_transform is not None:
            spec = self.spec_transform(spec)
        
#         return spec, scene_encoded
        return {"spec": spec, "wave": wave}, scene_encoded


class Task1BEvaluationDataset(data.Dataset):
    """
    
    """
    
    def __init__(self, data_dir, label_list):
        self.data_arr = pd.read_csv(data_dir, sep="\t").to_dict('records')
        self.data_dir = data_dir
        self.label_list = label_list
        self.audio_dir = "/mnt/ssd/data/tau_audiovisual_evaluation"
        
    def __len__(self):
        return len(self.data_arr)
    
    def __getitem__(self, idx):
        item = self.data_arr[idx]
        item_filepath = item['filename_audio']
        item_filename = item['filename_audio'].replace("audio/", "")
        
        
        audio, sr = librosa.load(os.path.join(self.audio_dir, item_filepath), sr=48000)
        padded = np.zeros(48000, dtype='float32')
        wave = audio[:48000]
        padded[0:len(wave)] = wave
        
        spec = librosa.feature.melspectrogram(padded, n_fft=2048, hop_length=256, n_mels=128, sr=48000, fmin=0, fmax=24000)
        
        spec = np.log(spec)
#         print(spec)

        # z-score normalization
        std = spec.std()
        mean = spec.mean()
        spec = (spec - mean) / std
        
        
#         return spec, scene_encoded
        return {"spec": spec, "wave": padded}, item_filename