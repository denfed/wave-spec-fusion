import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch
import wandb
# import leaf_audio_pytorch.frontend as frontend
# from leaf_audio_pytorch.postprocessing import log_compression
from models.modules.fusion import MFB
from models.modules.sincnet.dnn_models import *
# from leaf_audio_pytorch.convolution import GaborConv1D
# from leaf_audio_pytorch.frontend import SquaredModulus
# import leaf_audio_pytorch.initializers as initializers

    
class SpectrogramClassifier(BaseModel):
    def __init__(self,
                 n_classes,
                 n_in_channels=1,
                 non_linearity='LeakyReLU',
                 dropout=0.3,
                 latent_size=2048,
                 cn_feature_n=[32, 64, 128, 256, 512],
                 kernel_size=3,
                 max_pool_kernel=(2,2),
                 fc_layer_n=[1024, 512],
                 use_leaf=False,
                 leaf_sample_rate=48000,
                 leaf_window_len=42,
                 leaf_window_stride=40):
        super().__init__()
        
#         test_input_spectrogram = torch.zeros(3, n_in_channels, n_bins, n_frames)
        self.use_leaf = use_leaf
    
        if self.use_leaf:
            self.leaf = frontend.Leaf(n_filters=40, sample_rate=leaf_sample_rate, window_len=leaf_window_len, window_stride=leaf_window_stride)
        
        cn = []
        
        for ilb, n_out in enumerate(cn_feature_n):
            if ilb == 0:
                cn.append(nn.Conv2d(n_in_channels, n_out, kernel_size=kernel_size, padding=kernel_size//2))
                cn.append(nn.BatchNorm2d(n_out))
                cn.append(getattr(nn, non_linearity)())
                cn.append(nn.MaxPool2d(kernel_size=max_pool_kernel))
            else:
                cn.append(nn.Conv2d(
                    cn_feature_n[ilb-1], n_out, 
                    kernel_size=kernel_size, padding=kernel_size//2)
                ),
                cn.append(nn.BatchNorm2d(n_out))
                cn.append(getattr(nn, non_linearity)())
                cn.append(nn.MaxPool2d(kernel_size=max_pool_kernel))
                
        cn.append(nn.AdaptiveAvgPool2d((1,latent_size//cn_feature_n[-1])))
                    
        self.cnn = nn.Sequential(*cn)
        
#         _, cn_channels, cn_bins, cn_frames = self.cnn(test_input_spectrogram).shape
#         self.fc_in = cn_channels * cn_frames * cn_bins
        self.fc_in = latent_size
        
        fc = [
            nn.Flatten(),
            nn.Dropout(dropout)
        ]
        
        for il, n_out in enumerate(fc_layer_n):
            n_in = self.fc_in if il == 0 else fc_layer_n[il-1]
            fc.append(nn.Linear(n_in, n_out))
            fc.append(getattr(nn, non_linearity)())
            fc.append(nn.BatchNorm1d(n_out))
            fc.append(nn.Dropout(dropout))
        
        fc.append(nn.Linear(fc_layer_n[-1], n_classes))
        
        self.tail = nn.Sequential(*fc)
                    
    def forward(self, x):
        x = x['spec']
        
        if self.use_leaf:
            x = x.unsqueeze(1)
            x = self.leaf(x)
        
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = self.tail(x)
        return x
    
    
class WaveformClassifier(BaseModel):
    def __init__(self, 
                 input_length, 
                 num_classes=12,
                 n_in_channels=1,
                 non_linearity='LeakyReLU',
                 dropout=0.3,
                 latent_size=2048,
                 cn_feature_n=[32, 64, 128, 256, 512],
                 max_pool_kernel=8,
                 kernel_0_size=7,
                 kernel_size=7,
                 fc_layer_n=[1024,512]):
        
        super().__init__()
        test_input_waveform = torch.zeros(3, n_in_channels, input_length)
        
        cn = []
        for ilb, n_out in enumerate(cn_feature_n):
            if ilb == 0:
                cn.append(nn.Conv1d(n_in_channels, n_out, kernel_size=kernel_0_size, padding=kernel_size//2))
                cn.append(nn.BatchNorm1d(n_out))
                cn.append(getattr(nn, non_linearity)())
                cn.append(nn.MaxPool1d(kernel_size=max_pool_kernel))
            else:
                cn.append(nn.Conv1d(
                    cn_feature_n[ilb-1], n_out, 
                    kernel_size=kernel_size, padding=kernel_size // 2)
                ),
                cn.append(nn.BatchNorm1d(n_out))
                cn.append(getattr(nn, non_linearity)())
                cn.append(nn.MaxPool1d(kernel_size=max_pool_kernel))
                
        cn.append(nn.AdaptiveAvgPool1d(latent_size//cn_feature_n[-1]))
            
        self.frontend = nn.Sequential(*cn)

        # Build the FC head
        _, cn_features, cn_slices = self.frontend(test_input_waveform).shape
        
        self.fc_in = cn_features * cn_slices
        
        fc = [
            nn.Flatten(),
            nn.Dropout(dropout)
        ]
        
        for il, n_out in enumerate(fc_layer_n):
            n_in = self.fc_in if il == 0 else fc_layer_n[il-1]
            fc.append(nn.Linear(n_in, n_out))
            fc.append(getattr(nn, non_linearity)())
            fc.append(nn.BatchNorm1d(n_out))
            fc.append(nn.Dropout(dropout))
        
        fc.append(nn.Linear(fc_layer_n[-1], num_classes))
        
        self.tail = nn.Sequential(*fc)

    def forward(self, x):
        x = x['data']
        x = x.unsqueeze(1)
        x = self.frontend(x)
        x = self.tail(x)

        return x
    
    
class WaveformExtractor(nn.Module):
    def __init__(self,
                 n_in_channels=1,
                 non_linearity='LeakyReLU',
                 latent_size=2048,
                 parameterization='normal',
                 kernel_0_size=251,
                 cn_feature_n=[32, 64, 128, 256, 512],
                 max_pool_kernel=6,
                 kernel_size=7,
                 layernorm_fusion=False):
        
        super(WaveformExtractor, self).__init__()
#         test_input_waveform = torch.zeros(3, n_in_channels, input_length)
        
        cn = []
        for ilb, n_out in enumerate(cn_feature_n):
            if ilb == 0:
                if parameterization == 'normal':
                    cn.append(nn.Conv1d(n_in_channels, n_out, kernel_size=kernel_0_size, padding=kernel_0_size//2))
                elif parameterization == 'sinc':
                    cn.append(SincConv_fast(n_out, kernel_size=kernel_0_size, sample_rate=48000, padding=kernel_0_size//2))
                elif parameterization == 'leaf':
                    cn.append(GaborConv1D(n_out*2, kernel_size=kernel_0_size, strides=1, padding=kernel_0_size//2, use_bias=False, 
                                          input_shape=(None, None, 1), 
                                          kernel_initializer=initializers.GaborInit,
                                          kernel_regularizer=None,
                                          name='complex_conv',
                                          trainable=True))
                    cn.append(SquaredModulus())
                cn.append(nn.BatchNorm1d(n_out))
                cn.append(getattr(nn, non_linearity)())
                cn.append(nn.MaxPool1d(kernel_size=max_pool_kernel))
            else:
                cn.append(nn.Conv1d(
                    cn_feature_n[ilb-1], n_out, 
                    kernel_size=kernel_size, padding=kernel_size // 2)
                ),
                cn.append(nn.BatchNorm1d(n_out))
                cn.append(getattr(nn, non_linearity)())
                cn.append(nn.MaxPool1d(kernel_size=max_pool_kernel))
                
        cn.append(nn.AdaptiveAvgPool1d(latent_size//cn_feature_n[-1]))
        
        if layernorm_fusion:
            cn.append(nn.Flatten())
            cn.append(nn.LayerNorm(latent_size))
            
        self.frontend = nn.Sequential(*cn)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.frontend(x)
        return x
    
    
class SpectrogramExtractor(nn.Module):
    def __init__(self,
                 n_in_channels=1,
                 non_linearity='LeakyReLU',
                 latent_size=2048,
                 cn_feature_n=[32, 64, 128, 256, 512],
                 kernel_size=3,
                 max_pool_kernel=(2,2),
                 layernorm_fusion=False):
        super(SpectrogramExtractor, self).__init__()
        
#         test_input_spectrogram = torch.zeros(3, n_in_channels, n_bins, n_frames)
        
        cn = []
        
        for ilb, n_out in enumerate(cn_feature_n):
            if ilb == 0:
                cn.append(nn.Conv2d(n_in_channels, n_out, kernel_size=kernel_size, padding=kernel_size//2))
                cn.append(nn.BatchNorm2d(n_out))
                cn.append(getattr(nn, non_linearity)())
                cn.append(nn.MaxPool2d(kernel_size=max_pool_kernel))
            else:
                cn.append(nn.Conv2d(
                    cn_feature_n[ilb-1], n_out, 
                    kernel_size=kernel_size, padding=kernel_size//2)
                ),
                cn.append(nn.BatchNorm2d(n_out))
                cn.append(getattr(nn, non_linearity)())
                cn.append(nn.MaxPool2d(kernel_size=max_pool_kernel))
                
        cn.append(nn.AdaptiveAvgPool2d((1,latent_size//cn_feature_n[-1])))
        
        if layernorm_fusion:
            cn.append(nn.Flatten())
            cn.append(nn.LayerNorm(latent_size))
                    
        self.cnn = nn.Sequential(*cn)
                    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        return x
    
    
class MultiModalFusionClassifier(BaseModel):
    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.0)
            m.bias.data.fill_(0.0)
    
    def __init__(self,
                 input_length,
                 n_bins,
                 n_frames,
                 num_classes,
                 fusion_method='sum',
                 parameterization='normal',
                 non_linearity='ReLU',
                 dropout=0.3,
                 fc_layer_n=[1024,512]):
        super().__init__()
        
        self.fusion_method = fusion_method
        
        self.wave = WaveformExtractor(parameterization=parameterization)
        self.spec = SpectrogramExtractor(1)
        
        test_input_waveform = torch.zeros(3, input_length)
        test_input_spectrogram = torch.zeros(3, n_bins, n_frames)
        
        # Build the FC head
        _, w_cn_features, w_cn_slices = self.wave(test_input_waveform).shape
        self.wave_fc_in = w_cn_features * w_cn_slices
        _, s_cn_channels, s_cn_bins, s_cn_frames = self.spec(test_input_spectrogram).shape
        self.spec_fc_in = s_cn_channels * s_cn_frames * s_cn_bins
        
        assert self.wave_fc_in == self.spec_fc_in
        
        self.flat = nn.Flatten()
        
        fc = [
            nn.Dropout(dropout)
        ]
        
        if fusion_method == 'concat':
            self.wave_fc_in = self.wave_fc_in + self.spec_fc_in
            
        if fusion_method == 'mfb':
            self.mfb = MFB(self.wave_fc_in, self.spec_fc_in)
            
        if fusion_method == 'sum-attention-noinit':
            self.attention = nn.Sequential(
                nn.Linear(self.wave_fc_in*2, self.wave_fc_in*4),
                getattr(nn, non_linearity)(),
                nn.Linear(self.wave_fc_in*4, self.wave_fc_in//2),
                getattr(nn, non_linearity)(),
                nn.Linear(self.wave_fc_in//2, 2),
                nn.Softmax(dim=1)
            )
        if fusion_method == 'sum-attention-init':
            self.attention = nn.Sequential(
                nn.Linear(self.wave_fc_in*2, self.wave_fc_in*4),
                getattr(nn, non_linearity)(),
                nn.Linear(self.wave_fc_in*4, self.wave_fc_in//2),
                getattr(nn, non_linearity)(),
                nn.Linear(self.wave_fc_in//2, 2),
                nn.Softmax(dim=1)
            )
            self.init_weights(self.attention)
        
        for il, n_out in enumerate(fc_layer_n):
            n_in = self.wave_fc_in if il == 0 else fc_layer_n[il-1]
            fc.append(nn.Linear(n_in, n_out))
            fc.append(getattr(nn, non_linearity)())
            fc.append(nn.BatchNorm1d(n_out))
            fc.append(nn.Dropout(dropout))
        
        fc.append(nn.Linear(fc_layer_n[-1], num_classes))
        
        self.tail = nn.Sequential(*fc)
        
    def forward(self, x):
        w = x['wave']
        s = x['spec']
        w = self.wave(w)
        s = self.spec(s)
        
        w_flat = self.flat(w)
        s_flat = self.flat(s)
        
        # Select fusion method
        if self.fusion_method == 'sum':
            combined_features = w_flat.add(s_flat)
        elif self.fusion_method == 'concat':
            combined_features = torch.cat((w_flat, s_flat), dim=1)
        elif self.fusion_method == 'mfb':
            combined_features, _ = self.mfb(w_flat.unsqueeze(1), s_flat.unsqueeze(1))
            combined_features = combined_features.squeeze(1)
        elif self.fusion_method == 'sum-attention-noinit' or self.fusion_method == 'sum-attention-init':
            concat_features = torch.cat((w_flat, s_flat), dim=1)
            att = self.attention(concat_features)
        
            att_1, att_2 = torch.split(att, 1, dim=1)

            combined_features = (w_flat*att_1).add((s_flat*att_2))

        res = self.tail(combined_features)
        
        return res
    
    
class LeafClassifier(BaseModel):
    def __init__(self,
                 n_classes,
                 learn_pooling=False,
                 learn_filters=False,
                 n_in_channels=1,
                 non_linearity='LeakyReLU',
                 dropout=0.3,
                 latent_size=1024,
                 cn_feature_n=[32, 64, 128, 256, 512],
                 kernel_size=3,
                 max_pool_kernel=(2,2),
                 fc_layer_n=[256]):
        super().__init__()
        
#         test_input_spectrogram = torch.zeros(3, n_in_channels, n_bins, n_frames)

        self.leaf = frontend.Leaf(n_filters=40, sample_rate=48000, window_len=42, window_stride=40, learn_pooling=learn_pooling, learn_filters=learn_filters, compression_fn=log_compression)
        
        cn = []
        
        for ilb, n_out in enumerate(cn_feature_n):
            if ilb == 0:
                cn.append(nn.Conv2d(n_in_channels, n_out, kernel_size=kernel_size, padding=kernel_size//2))
                cn.append(nn.BatchNorm2d(n_out))
                cn.append(getattr(nn, non_linearity)())
                cn.append(nn.MaxPool2d(kernel_size=max_pool_kernel))
            else:
                cn.append(nn.Conv2d(
                    cn_feature_n[ilb-1], n_out, 
                    kernel_size=kernel_size, padding=kernel_size//2)
                ),
                cn.append(nn.BatchNorm2d(n_out))
                cn.append(getattr(nn, non_linearity)())
                cn.append(nn.MaxPool2d(kernel_size=max_pool_kernel))
                
        cn.append(nn.AdaptiveAvgPool2d((1,latent_size//cn_feature_n[-1])))
                    
        self.cnn = nn.Sequential(*cn)
        
#         _, cn_channels, cn_bins, cn_frames = self.cnn(test_input_spectrogram).shape
#         self.fc_in = cn_channels * cn_frames * cn_bins
        self.fc_in = latent_size
        
        fc = [
            nn.Flatten(),
            nn.Dropout(dropout)
        ]
        
        for il, n_out in enumerate(fc_layer_n):
            n_in = self.fc_in if il == 0 else fc_layer_n[il-1]
            fc.append(nn.Linear(n_in, n_out))
            fc.append(getattr(nn, non_linearity)())
            fc.append(nn.BatchNorm1d(n_out))
            fc.append(nn.Dropout(dropout))
        
        fc.append(nn.Linear(fc_layer_n[-1], n_classes))
        
        self.tail = nn.Sequential(*fc)
                    
    def forward(self, x):
        x = x['data']
        x = x.unsqueeze(1)
        x = self.leaf(x)
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = self.tail(x)
        return x
    
    
class WaveformParameterizedClassifier(BaseModel):
    def __init__(self, 
                 input_length, 
                 num_classes=12,
                 parameterization='normal',  # 'normal', 'sinc', 'leaf'
                 n_in_channels=1,
                 non_linearity='LeakyReLU',
                 dropout=0.3,
                 latent_size=2048,
                 cn_feature_n=[32, 64, 128, 256, 512],
                 max_pool_kernel=8,
                 kernel_0_size=7,
                 kernel_size=7,
                 fc_layer_n=[1024,512]):
        
        super().__init__()
        test_input_waveform = torch.zeros(3, n_in_channels, input_length)
        
        cn = []
        for ilb, n_out in enumerate(cn_feature_n):
            if ilb == 0:
                if parameterization == 'normal':
                    cn.append(nn.Conv1d(n_in_channels, n_out, kernel_size=kernel_0_size, padding=kernel_0_size//2))
                elif parameterization == 'sinc':
                    cn.append(SincConv_fast(n_out, kernel_size=kernel_0_size, sample_rate=48000, padding=kernel_0_size//2))
                elif parameterization == 'leaf':
                    cn.append(GaborConv1D(n_out*2, kernel_size=kernel_0_size, strides=1, padding=kernel_0_size//2, use_bias=False, 
                                          input_shape=(None, None, 1), 
                                          kernel_initializer=initializers.GaborInit,
                                          kernel_regularizer=None,
                                          name='complex_conv',
                                          trainable=True))
                    cn.append(SquaredModulus())
                cn.append(nn.BatchNorm1d(n_out))
                cn.append(getattr(nn, non_linearity)())
                cn.append(nn.MaxPool1d(kernel_size=max_pool_kernel))
            else:
                cn.append(nn.Conv1d(
                    cn_feature_n[ilb-1], n_out, 
                    kernel_size=kernel_size, padding=kernel_size // 2)
                ),
                cn.append(nn.BatchNorm1d(n_out))
                cn.append(getattr(nn, non_linearity)())
                cn.append(nn.MaxPool1d(kernel_size=max_pool_kernel))
                
        cn.append(nn.AdaptiveAvgPool1d(latent_size//cn_feature_n[-1]))
            
        self.frontend = nn.Sequential(*cn)

        # Build the FC head
        _, cn_features, cn_slices = self.frontend(test_input_waveform).shape
        
        self.fc_in = cn_features * cn_slices
        
        fc = [
            nn.Flatten(),
            nn.Dropout(dropout)
        ]
        
        for il, n_out in enumerate(fc_layer_n):
            n_in = self.fc_in if il == 0 else fc_layer_n[il-1]
            fc.append(nn.Linear(n_in, n_out))
            fc.append(getattr(nn, non_linearity)())
            fc.append(nn.BatchNorm1d(n_out))
            fc.append(nn.Dropout(dropout))
        
        fc.append(nn.Linear(fc_layer_n[-1], num_classes))
        
        self.tail = nn.Sequential(*fc)

    def forward(self, x):
        x = x['wave']
        x = x.unsqueeze(1)
        x = self.frontend(x)
        x = self.tail(x)
#         print(self.frontend[0].low_hz_)

        return x
    
    
class SmallMultiModalFusionClassifier(BaseModel):
    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.0)
            m.bias.data.fill_(0.0)
    
    def __init__(self,
                 input_length,
                 n_bins,
                 n_frames,
                 num_classes,
                 fusion_method='sum',
                 parameterization='normal',
                 non_linearity='LeakyReLU',
                 layernorm_fusion=False,
                 dropout=0.3,
                 fc_layer_n=[512,256],
                 kernel_0_size=251):
        super().__init__()
        
        self.fusion_method = fusion_method
        self.layernorm_fusion = layernorm_fusion
        
        self.wave = WaveformExtractor(
                 n_in_channels=1,
                 non_linearity='LeakyReLU',
                 latent_size=1024,
                 parameterization=parameterization,
                 kernel_0_size=kernel_0_size,
                 cn_feature_n=[32, 64, 128, 256],
                 max_pool_kernel=8,
                 kernel_size=7,
                 layernorm_fusion=layernorm_fusion)
        self.spec = SpectrogramExtractor(
                 n_in_channels=1,
                 non_linearity='LeakyReLU',
                 latent_size=1024,
                 cn_feature_n=[32, 64, 128, 256],
                 kernel_size=3,
                 max_pool_kernel=(2,2),
                 layernorm_fusion=layernorm_fusion)
        
        test_input_waveform = torch.zeros(3, input_length)
        test_input_spectrogram = torch.zeros(3, n_bins, n_frames)
        
        # Build the FC head
#         print(self.wave(test_input_waveform).shape)
        _, w_cn_features, w_cn_slices = self.wave(test_input_waveform).shape
        self.wave_fc_in = w_cn_features * w_cn_slices
        _, s_cn_channels, s_cn_bins, s_cn_frames = self.spec(test_input_spectrogram).shape
        self.spec_fc_in = s_cn_channels * s_cn_frames * s_cn_bins
        
        assert self.wave_fc_in == self.spec_fc_in
        
        self.flat = nn.Flatten()
        
        fc = [
            nn.Dropout(dropout)
        ]
        
        if fusion_method == 'concat':
            self.wave_fc_in = self.wave_fc_in + self.spec_fc_in
            
        if fusion_method == 'mfb':
            self.mfb = MFB(self.wave_fc_in, self.spec_fc_in, MFB_O=self.wave_fc_in, MFB_K=3)
            
        if fusion_method == 'sum-attention-noinit':
            self.attention = nn.Sequential(
                nn.Linear(self.wave_fc_in*2, self.wave_fc_in*4),
                getattr(nn, non_linearity)(),
                nn.Linear(self.wave_fc_in*4, self.wave_fc_in//2),
                getattr(nn, non_linearity)(),
                nn.Linear(self.wave_fc_in//2, 2),
                nn.Softmax(dim=1)
            )
        if fusion_method == 'sum-attention-init':
            self.attention = nn.Sequential(
                nn.Linear(self.wave_fc_in*2, self.wave_fc_in*4),
                getattr(nn, non_linearity)(),
                nn.Linear(self.wave_fc_in*4, self.wave_fc_in//2),
                getattr(nn, non_linearity)(),
                nn.Linear(self.wave_fc_in//2, 2),
                nn.Softmax(dim=1)
            )
            self.init_weights(self.attention)
        
        for il, n_out in enumerate(fc_layer_n):
            n_in = self.wave_fc_in if il == 0 else fc_layer_n[il-1]
            fc.append(nn.Linear(n_in, n_out))
            fc.append(getattr(nn, non_linearity)())
            fc.append(nn.BatchNorm1d(n_out))
            fc.append(nn.Dropout(dropout))
        
        fc.append(nn.Linear(fc_layer_n[-1], num_classes))
        
        self.tail = nn.Sequential(*fc)
        
    def forward(self, x):
        
        w = x['wave']
        s = x['spec']
        w = self.wave(w)
        s = self.spec(s)
        
#         print(self.wave.frontend[0].low_hz_)
        if not self.layernorm_fusion:
            w_flat = self.flat(w)
            s_flat = self.flat(s)
        else:
            w_flat = w
            s_flat = s
        
        # Select fusion method
        if self.fusion_method == 'sum':
            combined_features = w_flat.add(s_flat)
#             combined_features = w_flat
        elif self.fusion_method == 'concat':
            w_flat = F.normalize(w_flat, p=2.0, dim=1, eps=1e-12)
            s_flat = F.normalize(s_flat, p=2.0, dim=1, eps=1e-12)
            
            combined_features = torch.cat((w_flat, s_flat), dim=1)
        elif self.fusion_method == 'mfb':
            combined_features, _ = self.mfb(w_flat.unsqueeze(1), s_flat.unsqueeze(1))
            combined_features = combined_features.squeeze(1)
        elif self.fusion_method == 'sum-attention-noinit' or self.fusion_method == 'sum-attention-init':
            concat_features = torch.cat((w_flat, s_flat), dim=1)
            att = self.attention(concat_features)
        
            att_1, att_2 = torch.split(att, 1, dim=1)

            combined_features = (w_flat*att_1).add((s_flat*att_2))

        res = self.tail(combined_features)
        
        return res


class ScratchEnsemble(BaseModel):
    def __init__(self):
        super().__init__()
        self.spec_model_kwargs = {
                "n_classes": 10,
                "latent_size": 1024,
                "fc_layer_n": [512, 256],
                "cn_feature_n": [32, 64, 128, 256]
            }
        
        self.spec_model = SpectrogramClassifier(**self.spec_model_kwargs)
        
        self.wave_model_kwargs = {
                "input_length": 48000,
                "num_classes": 10,
                "kernel_0_size": 251,
                "parameterization": "sinc",
                "max_pool_kernel": 8,
                "latent_size": 1024,
                "cn_feature_n": [32, 64, 128, 256],
                "fc_layer_n": [512, 256]
            }
        
        self.wave_model = WaveformParameterizedClassifier(**self.wave_model_kwargs)
        
    def forward(self, x):
        self.wave = self.wave_model(x)
        
        self.spec = self.spec_model(x)
        
        output = torch.stack([self.wave, self.spec]).mean(dim=0)
        
        return output