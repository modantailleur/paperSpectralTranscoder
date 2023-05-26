#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 15:21:15 2022

@author: user
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.pinv_transcoder as pt
from pann.models import ResNet38Mels, ResNet38
from pathlib import Path
from yamnet.torch_audioset.yamnet.model import yamnet as torch_yamnet
from efficientnet_pytorch import EfficientNet

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(input.shape[0], *self.shape)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.shape[0], -1)


class FC(nn.Module):
    def __init__(self, scores_len, output_len, dtype=torch.FloatTensor):
        super().__init__()
        self.output_len = output_len
        self.scores_shape = scores_len
        self.fc = nn.Linear(scores_len, output_len)
        self.input_fc = nn.Linear(scores_len, 100)
        self.output_fc = nn.Linear(100, output_len)
        self.m = nn.Sigmoid()
        #self.fc = nn.Linear(scores_len, 3)

    def forward(self, x):
        
        #x = torch.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))

        #MLP version
        x_interm = self.input_fc(x)
        y_pred = self.output_fc(x_interm)
        y_pred = self.m(y_pred)

        #FC version
        # y_pred = self.fc(x)
        # y_pred = self.m(y_pred)

        return y_pred
    
class MLP(nn.Module):
    def __init__(self, input_shape, output_shape, dtype=torch.FloatTensor, 
                 hl_1=300, hl_2=3000):
        super().__init__()
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hl_1 = hl_1
        self.hl_2 = hl_2
        self.input_fc = nn.Linear(input_shape[0]*input_shape[1], hl_1)
        self.hidden_fc = nn.Linear(hl_1, hl_2)
        self.output_fc = nn.Linear(hl_2, output_shape[0]*output_shape[1])
        self.dtype = dtype

    def forward(self, x):

        # x = [batch size, height, width]

        # x = [batch size, height * width]
        x = torch.reshape(x, (x.shape[0], self.input_shape[0]*self.input_shape[1]))

        h_1 = F.relu(self.input_fc(x))

        h_2 = F.relu(self.hidden_fc(h_1))

        y_pred = self.output_fc(h_2)
        
        y_pred = torch.reshape(y_pred, (y_pred.shape[0], self.output_shape[0], self.output_shape[1]))

        return y_pred

class MLPPINV(nn.Module):
    def __init__(self, input_shape, output_shape, tho_tr, mels_tr, hl_1=300, 
                hl_2=3000, dtype=torch.FloatTensor, device=torch.device("cpu"),
                residual=True, interpolate=True, input_is_db=True):
        """
        Initializes the MLPPINV nn model class.

        Args:
        - input_shape: The shape of the input ((8, 29) for 1-s third-octave spectrograms).
        - output_shape: The shape of the output ((101, 64) for 1-s PANN Mel Spectrograms)
        - tho_tr: The third-octave transform used for converting audio into third-octave spectrograms
        - mels_tr: The Mel transform used for converting audio into Mel spectrograms.
        - hl_1: The number of hidden units in the first hidden layer (default: 300).
        - hl_2: The number of hidden units in the second hidden layer (default: 3000).
        - dtype: The data type for the model (default: torch.FloatTensor).
        - device: The device to run the model on (default: torch.device("cpu")).
        - residual: Whether to add residual to the PINV spectrogram, or just recreate a new spectrogram(default: True).
        - interpolate: Whether to use interpolation on time axis (default: True).
        - input_is_db: Whether the input is in decibels (default: True).
        """

        super().__init__()
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hl_1 = hl_1
        self.hl_2 = hl_2

        self.input_is_db = input_is_db

        self.residual = residual 
        self.interpolate = interpolate

        if self.interpolate:
            self.input_fc = nn.Linear(output_shape[0]*output_shape[1], hl_1)
        else:
            self.input_fc = nn.Linear(input_shape[0]*input_shape[1], hl_1)

        self.hidden_fc = nn.Linear(hl_1, hl_2)
        self.output_fc = nn.Linear(hl_2, output_shape[0]*output_shape[1])
        self.dtype = dtype
        self.tho_tr = tho_tr
        self.mels_tr = mels_tr
        self.device = device

    def forward(self, x):
        # x = [batch size, height, width]

        # x = [batch size, height * width]
        if self.interpolate:
            y = pt.pinv(x, self.tho_tr, self.mels_tr, reshape=self.output_shape[0], device=self.device, input_is_db=self.input_is_db)
            y_fc = torch.reshape(y, (y.shape[0], self.output_shape[0]*self.output_shape[1]))

        else:
            y = pt.pinv(x, self.tho_tr, self.mels_tr, reshape=None, device=self.device, input_is_db=self.input_is_db)
            y_fc = torch.reshape(y, (y.shape[0], self.input_shape[0]*self.input_shape[1]))

        h_1 = F.relu(self.input_fc(y_fc))
        h_2 = F.relu(self.hidden_fc(h_1))

        y_pred = self.output_fc(h_2)
        
        y_pred = torch.reshape(y_pred, (y_pred.shape[0], self.output_shape[0], self.output_shape[1]))

        if self.residual:
            y_pred = y - y_pred

        return y_pred
    
class CNN(nn.Module):
    def __init__(self, input_shape, output_shape, tho_tr, mels_tr, kernel_size=5, nb_channels=64, nb_layers=3, dilation=1, dtype=torch.FloatTensor,
                device=torch.device("cpu"), residual=True, interpolate=True, input_is_db=True):
        """
        Initializes the CNN nn model class.

        Args:
        - input_shape: The shape of the input ((8, 29) for 1-s third-octave spectrograms).
        - output_shape: The shape of the output ((101, 64) for 1-s PANN Mel Spectrograms)
        - tho_tr: The third-octave transform used for converting audio into third-octave spectrograms
        - mels_tr: The Mel transform used for converting audio into Mel spectrograms.
        - kernel_size: The size of the convolutional kernel (default: 5).
        - nb_channels: The number of channels in the convolutional layers (default: 64).
        - nb_layers: The number of convolutional layers (default: 3).
        - dilation: The dilation rate for the convolutional layers (default: 1).
        - dtype: The data type for the model (default: torch.FloatTensor).
        - device: The device to run the model on (default: torch.device("cpu")).
        - residual: Whether to add residual to the PINV spectrogram, or just recreate a new spectrogram(default: True).
        - interpolate: Whether to use interpolation on time axis (default: True).
        - input_is_db: Whether the input is in decibels (default: True).
        """
        super(CNN, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        
        self.kernel_size = kernel_size
        self.nb_channels = nb_channels
        self.nb_layers = nb_layers
        self.dilation = dilation
        
        self.residual = residual 
        self.interpolate = interpolate
        self.input_is_db = input_is_db

        self.dtype = dtype

        self.tho_tr = tho_tr
        self.mels_tr = mels_tr
        
        padding_size = int((kernel_size-1)/2)
        
        # conv module
        layers_conv = nn.ModuleList()
        layers_conv.append(nn.ReplicationPad2d((padding_size, padding_size, 1, 1)))
        layers_conv.append(nn.Conv2d(1, nb_channels, (3, kernel_size), stride=1))
        layers_conv.append(nn.ReLU())
        dil = 1
        for l in range(nb_layers-2):
            if dilation > 1:
                dil = dilation
                padding_size = int(dil*(kernel_size-1)/2)
            layers_conv.append(nn.ReplicationPad2d(
                (padding_size, padding_size, 1, 1)))
            layers_conv.append(nn.Conv2d(nb_channels, nb_channels,
                          (3, kernel_size), stride=1, dilation=(1, dil)))
            layers_conv.append(nn.ReLU())
        padding_size = int((kernel_size-1)/2)
        layers_conv.append(nn.ReplicationPad2d((padding_size, padding_size, 1, 1)))
        layers_conv.append(nn.Conv2d(nb_channels, 1, (3, kernel_size), stride=1))
        #MT: removed ReLU for converge issues
        #layers_conv.append(nn.ReLU())
        self.mod_conv = nn.Sequential(*layers_conv)

        self.device=device

    def forward(self, x):

        if self.interpolate:
            x = pt.pinv(x, self.tho_tr, self.mels_tr, reshape=self.output_shape[0], device=self.device, input_is_db=self.input_is_db)
        else:
            x = pt.pinv(x, self.tho_tr, self.mels_tr, reshape=None, device=self.device, input_is_db=self.input_is_db)

        # x = [batch size, height * width]

        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        if self.interpolate:
            y_fc = x
        else:
            # y_pred = [batch size, output dim]
            y_fc = self.mod_fc(x)

        # y_pred = [batch size, 1, height, width]
        y_fc = torch.reshape(y_fc, (batch_size, 1, self.output_shape[0], self.output_shape[1]))

        y_pred = self.mod_conv(y_fc)
        
        if self.residual:
            y_pred = y_fc - y_pred

        y_pred = y_pred.squeeze(dim=1)

        return y_pred

class PANNPINV(nn.Module):
    def __init__(self, input_shape, output_shape, tho_tr, mels_tr, dtype=torch.FloatTensor, device=torch.device("cpu"),
                residual=True, interpolate=True, input_is_db=True):
        """
        Initializes the PANNPINV nn model class. This is the class used for the transcoder "self" in exp_train_model/main_doce_training.py,
        if PANN model is used as the base classifier. 

        Args:
        - input_shape: The shape of the input ((8, 29) for 1-s third-octave spectrograms).
        - output_shape: The shape of the output ((101, 64) for 1-s PANN Mel Spectrograms)
        - tho_tr: The third-octave transform used for converting audio into third-octave spectrograms
        - mels_tr: The Mel transform used for converting audio into Mel spectrograms.
        - dtype: The data type for the model (default: torch.FloatTensor).
        - device: The device to run the model on (default: torch.device("cpu")).
        - residual: Whether to add residual to the PINV spectrogram, or just recreate a new spectrogram(default: True).
        - interpolate: Whether to use interpolation on time axis (default: True).
        - input_is_db: Whether the input is in decibels (default: True).
        """
        super().__init__()
        
        #PANN CNN model that takes Mel spectrogram as input
        # self.model = Cnn14_DecisionLevelMaxMels(sample_rate=mels_tr.sample_rate, window_size=mels_tr.window_size, 
        #     hop_size=mels_tr.hop_size, mel_bins=mels_tr.mel_bins, fmin=mels_tr.fmin, fmax=mels_tr.fmax, 
        #     classes_num=527)
        self.model = ResNet38Mels(sample_rate=mels_tr.sample_rate, window_size=mels_tr.window_size, 
            hop_size=mels_tr.hop_size, mel_bins=mels_tr.mel_bins, fmin=mels_tr.fmin, fmax=mels_tr.fmax, 
            classes_num=527)
            
        
        #PANN CNN model that takes audio as input
        # self.full_model = Cnn14_DecisionLevelMax(sample_rate=mels_tr.sample_rate, window_size=mels_tr.window_size, 
        #     hop_size=mels_tr.hop_size, mel_bins=mels_tr.mel_bins, fmin=mels_tr.fmin, fmax=mels_tr.fmax, 
        #     classes_num=527)
        self.full_model =  ResNet38(sample_rate=mels_tr.sample_rate, window_size=mels_tr.window_size, 
            hop_size=mels_tr.hop_size, mel_bins=mels_tr.mel_bins, fmin=mels_tr.fmin, fmax=mels_tr.fmax, 
            classes_num=527)
        
        ###############
        #models loading
        #checkpoint_path = Path().absolute() / 'pann' / 'Cnn14_DecisionLevelMax_mAP=0.385.pth'
        checkpoint_path = Path().absolute() / 'pann' / 'ResNet38_mAP=0.434.pth'

        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.full_model.load_state_dict(checkpoint['model'])
        
        full_model_dict = self.full_model.state_dict()
        model_dict = self.model.state_dict()
        
        # filter out unnecessary keys
        full_model_dict = {k: v for k, v in full_model_dict.items() if k in model_dict}
        # overwrite entries in the existing state dict
        model_dict.update(full_model_dict) 
        # load the new state dict
        self.model.load_state_dict(full_model_dict)
        self.model.to(device)


        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_is_db = input_is_db

        self.residual = residual 
        self.interpolate = interpolate

        self.dtype = dtype
        self.tho_tr = tho_tr
        self.mels_tr = mels_tr
        self.device = device

    def forward(self, x):
        y_fc = pt.pinv(x, self.tho_tr, self.mels_tr, reshape=self.output_shape[0], device=self.device, input_is_db=self.input_is_db)
        y_fc = torch.unsqueeze(y_fc, 1)
        y_pred = self.model(y_fc)['clipwise_output']
        return y_pred

class YAMNETPINV(nn.Module):
    def __init__(self, input_shape, output_shape, tho_tr, mels_tr, dtype=torch.FloatTensor, device=torch.device("cpu"),
                residual=True, interpolate=True, input_is_db=True):
        """
        Initializes the YAMNETPINV nn model class. This is the class used for the transcoder "self" in exp_train_model/main_doce_training.py,
        if YamNet model is used as the base classifier. 

        Args:
        - input_shape: The shape of the input ((8, 29) for 1-s third-octave spectrograms).
        - output_shape: The shape of the output ((101, 64) for 1-s PANN Mel Spectrograms)
        - tho_tr: The third-octave transform used for converting audio into third-octave spectrograms
        - mels_tr: The Mel transform used for converting audio into Mel spectrograms.
        - dtype: The data type for the model (default: torch.FloatTensor).
        - device: The device to run the model on (default: torch.device("cpu")).
        - residual: Whether to add residual to the PINV spectrogram, or just recreate a new spectrogram(default: True).
        - interpolate: Whether to use interpolation on time axis (default: True).
        - input_is_db: Whether the input is in decibels (default: True).
        """

        super().__init__()
        
        ###############
        #models loading
        self.model = torch_yamnet(pretrained=False)
        # Manually download the `yamnet.pth` file.
        self.model.load_state_dict(torch.load(Path().absolute() / 'yamnet' / 'yamnet.pth', map_location=device))
        self.model.to(device)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_is_db = input_is_db

        self.residual = residual 
        self.interpolate = interpolate

        self.dtype = dtype
        self.tho_tr = tho_tr
        self.mels_tr = mels_tr
        self.device = device

    def forward(self, x):

        y_fc = pt.pinv(x, self.tho_tr, self.mels_tr, reshape=self.output_shape[0], device=self.device, input_is_db=self.input_is_db)
        y_fc = torch.unsqueeze(y_fc, 1)
        y_pred = self.model(y_fc, to_prob=True)
        return y_pred
    

class EffNet(nn.Module):
    def __init__(self, mels_tr, effnet_type, dtype=torch.FloatTensor, device=torch.device("cpu")):
        """
        Initializes the EffNet nn model class. This is the class used for the transcoders effnet_b0 and effnet_b7 in exp_train_model/main_doce_training.py. 

        Args:
        - mels_tr: The Mel transform used for converting audio into Mel spectrograms. Here it just serves to retrieve the number of labels
                    that corresponds to the classifier outputs (527 for PANN, 521 for YamNet)
        - effnet_type: effnet_b0 or effnet_b7
        - dtype: The data type for the model (default: torch.FloatTensor).
        - device: The device to run the model on (default: torch.device("cpu")).
        """
        super().__init__()
        
        ###############
        #models loading
        if effnet_type == "effnet_b0":
            self.model = EfficientNet.from_name('efficientnet-b0', num_classes=mels_tr.n_labels)
            state_dict = torch.load("./efficient_net/efficientnet-b0-355c32eb.pth")
            state_dict.pop('_fc.weight')
            state_dict.pop('_fc.bias')
            self.model.load_state_dict(state_dict, strict=False)

            # modify input conv layer to accept 1x101x64 input
            self.model._conv_stem = nn.Conv2d(1, 32, kernel_size=3, stride=2, bias=False)

        if effnet_type == "effnet_b7":
            self.model = EfficientNet.from_name('efficientnet-b7', num_classes=mels_tr.n_labels)
            state_dict = torch.load("./efficient_net/efficientnet-b7-dcc49843.pth")
            state_dict.pop('_fc.weight')
            state_dict.pop('_fc.bias')
            self.model.load_state_dict(state_dict, strict=False)

            # modify input conv layer to accept 1x101x64 input
            self.model._conv_stem = nn.Conv2d(1, 64, kernel_size=3, stride=2, bias=False)

        self.model.to(device)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = F.interpolate(x, size=(101, 64), mode='nearest')
        y_pred = self.model(x)
        #clamp gave better results than sigmoid function
        #y_pred = torch.sigmoid(y_pred)
        y_pred = torch.clamp(y_pred, min=0, max=1)
        return y_pred

# class CNN_GEN(nn.Module):
#     def __init__(self, input_shape, output_shape, tho_tr, mels_tr, kernel_size=5, nb_channels=64, nb_layers=3, dilation=0, dtype=torch.FloatTensor,
#                 device=torch.device("cpu")):
#         super(CNN_GEN, self).__init__()
#         #input_shape: (8, 64)
#         self.input_shape = input_shape
#         #output_shape: (101, 64)
#         self.output_shape = output_shape
        
#         self.kernel_size = kernel_size
#         self.nb_channels = nb_channels
#         self.nb_layers = nb_layers
#         self.dilation = dilation
        
#         self.dtype = dtype
#         #for pinv
#         self.tho_tr = tho_tr
#         self.mels_tr = mels_tr
        
#         input_shape_flatten = input_shape[0]*input_shape[1]
#         output_shape_flatten = output_shape[0]*output_shape[1]
        
#         padding_size = int((kernel_size-1)/2)
        
#         # fully connected module
#         layers_fc = nn.ModuleList()
#         layers_fc.append(nn.Linear(input_shape_flatten, output_shape_flatten))
#         # MT: activate to add more layers
#         # layers_fc.append(nn.ReLU())
#         # layers_fc.append(nn.Linear(output_shape_flatten, output_shape_flatten))
#         self.mod_fc = nn.Sequential(*layers_fc)
        
#         # conv module
#         layers_conv = nn.ModuleList()
#         layers_conv.append(nn.ReplicationPad2d((padding_size, padding_size, 1, 1)))
#         layers_conv.append(nn.Conv2d(1, nb_channels, (3, kernel_size), stride=1))
#         layers_conv.append(nn.ReLU())
#         dil = 1
#         for l in range(nb_layers-2):
#             if dilation > 1:
#                 dil = dilation
#                 padding_size = int(dil*(kernel_size-1)/2)
#             layers_conv.append(nn.ReplicationPad2d(
#                 (padding_size, padding_size, 1, 1)))
#             layers_conv.append(nn.Conv2d(nb_channels, nb_channels,
#                           (3, kernel_size), stride=1, dilation=(1, dil)))
#             layers_conv.append(nn.ReLU())
#         padding_size = int((kernel_size-1)/2)
#         layers_conv.append(nn.ReplicationPad2d((padding_size, padding_size, 1, 1)))
#         layers_conv.append(nn.Conv2d(nb_channels, 1, (3, kernel_size), stride=1))
#         #MT: removed ReLU for converge issues
#         #layers_conv.append(nn.ReLU())
#         self.mod_conv = nn.Sequential(*layers_conv)

#         self.device=device

#     def forward(self, x):
        
#         x = pt.pinv(x, self.tho_tr, self.mels_tr, reshape=None, device=self.device)
        
#         # x = [batch size, height * width]
#         batch_size = x.shape[0]
#         x = x.view(batch_size, -1)

#         # y_pred = [batch size, output dim]
#         y_fc = self.mod_fc(x)

#         # y_pred = [batch size, 1, height, width]
#         #y_pred = torch.reshape(y_pred, (batch_size, 1, 101, 64))
#         y_fc = torch.reshape(y_fc, (batch_size, 1, self.output_shape[0], self.output_shape[1]))
        
#         print('AAAAAAAAAAA')
#         print(y_fc.shape)

#         y_pred = self.mod_conv(y_fc)
        
#         y_pred = y_pred.squeeze(dim=1)

#         return y_pred

#     def parameter_count(self):
#         par = list(self.parameters())
#         s = sum([np.prod(list(d.size())) for d in par])
#         return s

#     def parameter_print(self):
#         par = list(self.parameters())
#         for d in par:
#             print(d)