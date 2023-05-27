#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 10:05:35 2022

@author: user
"""
import os
import sys
# Add the parent directory of the project directory to the module search path
project_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_parent_dir)
import librosa
import numpy as np
import random
import numpy as np
import librosa
import numpy as np
import librosa
import torch.utils.data
import torch
from classif_utils import load_sonyc_meta, load_urbansound8k_meta
from utils.util import sort_labels_by_score
from transcoders import ThirdOctaveToMelTranscoderPinv, ThirdOctaveToMelTranscoder
import argparse

random.seed(0)
torch.manual_seed(0)

if torch.cuda.is_available():
    # Set the random seed for GPU (if available)
    torch.cuda.manual_seed(0)

def main(config):

    data_path = config.audio_dataset_path + "/" + config.audio_dataset_name
    save_data_path = config.output_path + "/" + config.audio_dataset_name + '-LOGITS'
    # Check if the path exists
    if not os.path.exists(save_data_path):
        # Create the directory recursively
        os.makedirs(save_data_path)

    MODEL_PATH = "./reference_models"
    cnn_pann_name = 'classifier=PANN+dataset=full+dilation=1+epoch=200+kernel_size=5+learning_rate=-3+nb_channels=64+nb_layers=5+prop_logit=100+step=train+transcoder=cnn_pinv+ts=1_model'
    cnn_yamnet_name = 'classifier=YamNet+dataset=full+dilation=1+epoch=200+kernel_size=5+learning_rate=-3+nb_channels=64+nb_layers=5+prop_logit=100+step=train+transcoder=cnn_pinv+ts=1_model'
    transcoder = 'cnn_pinv'
    dtype=torch.FloatTensor
    spec = True
    fs=32000
    force_cpu = False
    #manage gpu
    useCuda = torch.cuda.is_available() and not force_cpu

    if useCuda:
        print('Using CUDA.')
        dtype = torch.cuda.FloatTensor
        ltype = torch.cuda.LongTensor
        #MT: add
        device = torch.device("cuda:0")
    else:
        print('No CUDA available.')
        dtype = torch.FloatTensor
        ltype = torch.LongTensor
        #MT: add
        device = torch.device("cpu")

    if config.audio_dataset_name == "SONYC-UST":
        df, _, _, _ = load_sonyc_meta(data_path)

    if config.audio_dataset_name == "URBAN-SOUND-8K":
        df, _, _, _ = load_urbansound8k_meta(data_path, verbose=False)

    audio_dir = data_path + "/audio"
    len_df = len(df. index)

    #PANN
    transcoder_deep_pann = ThirdOctaveToMelTranscoder(transcoder, cnn_pann_name, MODEL_PATH, device)
    transcoder_pinv_pann = ThirdOctaveToMelTranscoderPinv(MODEL_PATH, cnn_pann_name, device)

    #YamNet
    transcoder_deep_yamnet = ThirdOctaveToMelTranscoder(transcoder, cnn_yamnet_name, MODEL_PATH, device)
    transcoder_pinv_yamnet = ThirdOctaveToMelTranscoderPinv(MODEL_PATH, cnn_yamnet_name, device)

    n_file = 0



    for f_audio in df.index:

        f_name = audio_dir + "/" + f_audio

        #######
        # PANN
        x_32k = librosa.load(f_name, sr=fs)[0]
        x_32k = librosa.util.normalize(x_32k)

        x_mels_pann_cnn = transcoder_deep_pann.transcode_from_wav_entire_file(x_32k)
        x_logit_pann_cnn = transcoder_deep_pann.mels_to_logit_entire_file(x_mels_pann_cnn, slice=False)
        x_mels_pann_gt = transcoder_deep_pann.mels_tr.wave_to_mels(x_32k)
        x_logit_pann_gt = transcoder_deep_pann.mels_to_logit_entire_file(x_mels_pann_gt, slice=False)

        # #######
        # YAMNET
        x_16k = librosa.load(f_name, sr=16000)[0]
        x_16k = librosa.util.normalize(x_16k)

        x_mels_yamnet_cnn, x_logit_yamnet_cnn = transcoder_deep_yamnet.transcode_from_wav(x_16k)
        x_logit_yamnet_cnn = np.mean(x_logit_yamnet_cnn, axis=1)
        x_logit_yamnet_cnn = x_logit_yamnet_cnn.reshape((x_logit_yamnet_cnn.shape[0], 1))

        #x_logit_yamnet_cnn = transcoder_deep_yamnet.mels_to_logit_entire_file(x_mels_yamnet_cnn)
        x_mels_yamnet_gt = transcoder_deep_yamnet.mels_tr.wave_to_mels(x_16k)
        x_logit_yamnet_gt = transcoder_deep_yamnet.mels_to_logit_sliced(x_mels_yamnet_gt)

        if config.verbose:
            print(f'XXXXXX FILE {f_audio} XXXXX')
            print('PANN')
            print('XXXXXXXXX original XXXXXXXXXXX')
            print(sort_labels_by_score(np.mean(x_logit_pann_gt, axis=1), transcoder_deep_pann.classif_inference.labels_str)[0][:10])
            print(sort_labels_by_score(np.mean(x_logit_pann_gt, axis=1), transcoder_deep_pann.classif_inference.labels_str)[1][:10])

            print('XXXXXXXXXXXX with transcoder XXXXXXXXXXXX')
            print(sort_labels_by_score(np.mean(x_logit_pann_cnn, axis=1), transcoder_deep_pann.classif_inference.labels_str)[0][:10])
            print(sort_labels_by_score(np.mean(x_logit_pann_cnn, axis=1), transcoder_deep_pann.classif_inference.labels_str)[1][:10])

            print('YamNet')
            print('XXXXXXXXX original XXXXXXXXXXX')
            print(sort_labels_by_score(np.mean(x_logit_yamnet_gt, axis=1), transcoder_deep_yamnet.classif_inference.labels_str)[0][:10])
            print(sort_labels_by_score(np.mean(x_logit_yamnet_gt, axis=1), transcoder_deep_yamnet.classif_inference.labels_str)[1][:10])

            print('XXXXXXXXXXXX with transcoder XXXXXXXXXXXX')
            print(sort_labels_by_score(np.mean(x_logit_yamnet_cnn, axis=1), transcoder_deep_yamnet.classif_inference.labels_str)[0][:10])
            print(sort_labels_by_score(np.mean(x_logit_yamnet_cnn, axis=1), transcoder_deep_yamnet.classif_inference.labels_str)[1][:10])

        np.save(save_data_path + "/" + (f_audio[:-4] + '_pann_cnn.npy'), x_logit_pann_cnn[:,0])
        np.save(save_data_path  + "/" + (f_audio[:-4] + '_pann_gt.npy'), x_logit_pann_gt[:,0])
        np.save(save_data_path  + "/" + (f_audio[:-4] + '_yamnet_cnn.npy'), x_logit_yamnet_cnn[:,0])
        np.save(save_data_path  + "/" + (f_audio[:-4] + '_yamnet_gt.npy'), x_logit_yamnet_gt[:,0])

        n_file += 1
        print('\r' + f'{n_file} / {len_df} files have been processed in dataset',end=' ')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate 1s Mels and Third-Octave spectrograms')

    parser.add_argument('--audio_dataset_path', type=str, default="./",
                        help='The path where the datasets are stored')
    parser.add_argument('--audio_dataset_name', type=str, default='URBAN-SOUND-8K',
                        help='The name of the dataset (URBAN-SOUND-8K or SONYC-UST)')
    parser.add_argument('--output_path', type=str, default='spectral_transcoder_data/data',
                        help='The name of the dataset')
    parser.add_argument('--verbose', type=bool, default=False, help='Show classification results for every file if set to True')
    config = parser.parse_args()
    main(config)