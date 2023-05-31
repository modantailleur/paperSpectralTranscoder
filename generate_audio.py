
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:02:54 2022

@author: user
"""

import numpy as np
import librosa
import numpy as np
import librosa
import os
import torch.utils.data
import torch
from scipy.io.wavfile import write
from transcoders import ThirdOctaveToMelTranscoderPinv, ThirdOctaveToMelTranscoder
from utils.util import sort_labels_by_score
import argparse

def main(config):
    MODEL_PATH = "./reference_models"
    filename = config.audio_file
    cnn_logits_name = 'classifier=PANN+dataset=full+dilation=1+epoch=200+kernel_size=5+learning_rate=-3+nb_channels=64+nb_layers=5+prop_logit=100+step=train+transcoder=cnn_pinv+ts=1_model'
    cnn_mels_name = 'classifier=PANN+dataset=full+dilation=1+epoch=200+kernel_size=5+learning_rate=-3+nb_channels=64+nb_layers=5+step=train+transcoder=cnn_pinv+ts=0_model'
    transcoder = 'cnn_pinv'
    dtype=torch.FloatTensor
    spec = True
    fs=32000
    full_filename = "audio/" + filename
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

    x_32k = librosa.load(full_filename, sr=fs)
    x_32k = np.array(x_32k[0])
    x_32k = librosa.util.normalize(x_32k)

    transcoder_cnn_logits_pann = ThirdOctaveToMelTranscoder(transcoder, cnn_logits_name, MODEL_PATH, device=device)
    transcoder_cnn_mels_pann = ThirdOctaveToMelTranscoder(transcoder, cnn_mels_name, MODEL_PATH, device=device)
    transcoder_pinv_pann = ThirdOctaveToMelTranscoderPinv(MODEL_PATH, cnn_logits_name, device, classifier="PANN")

    x_32k = librosa.load(full_filename, sr=fs)[0]
    x_32k = librosa.util.normalize(x_32k)

    x_mels_pann_cnn_logits = transcoder_cnn_logits_pann.transcode_from_wav_entire_file(x_32k)
    x_logit_pann_cnn_logits = transcoder_cnn_logits_pann.mels_to_logit_entire_file(x_mels_pann_cnn_logits, slice=False)
    x_mels_pann_gt = transcoder_cnn_logits_pann.mels_tr.wave_to_mels(x_32k)
    x_logit_pann_gt = transcoder_cnn_logits_pann.mels_to_logit_entire_file(x_mels_pann_gt, slice=False)
    x_mels_pann_pinv = transcoder_pinv_pann.transcode_from_wav(x_32k)
    x_mels_pann_cnn_mels = transcoder_cnn_mels_pann.transcode_from_wav_entire_file(x_32k)

    print('\n XXXXXXXXX ORIGINAL PANN CLASSIFIER (MEL INPUT) XXXXXXXXXXX')
    labels = sort_labels_by_score(np.mean(x_logit_pann_gt, axis=1), transcoder_cnn_logits_pann.classif_inference.labels_str)[1][:10]
    scores = sort_labels_by_score(np.mean(x_logit_pann_gt, axis=1), transcoder_cnn_logits_pann.classif_inference.labels_str)[0][:10]
    for k in range(len(labels)):
        print(f'{labels[k]} : {round(float(scores[k]), 2)}')

    print('\n XXXXXXXXXXXX TRANSCODED PANN CLASSIFIER (THIRD-OCTAVE INPUT) USING CNN-LOGITS TRANSCODER XXXXXXXXXXXX')
    labels = sort_labels_by_score(np.mean(x_logit_pann_cnn_logits, axis=1), transcoder_cnn_logits_pann.classif_inference.labels_str)[1][:10]
    scores = sort_labels_by_score(np.mean(x_logit_pann_cnn_logits, axis=1), transcoder_cnn_logits_pann.classif_inference.labels_str)[0][:10]
    for k in range(len(labels)):
        print(f'{labels[k]} : {round(float(scores[k]), 2)}')

    input_gen_cnn_logits = x_mels_pann_cnn_logits
    input_gen_cnn_mels = x_mels_pann_cnn_mels
    input_gen_pinv = x_mels_pann_pinv
    input = x_mels_pann_gt

    y_gen_cnn_logits = librosa.feature.inverse.mel_to_audio(input_gen_cnn_logits, sr=32000, n_fft=1024, hop_length=320, win_length=1024)
    y_gen_cnn_mels = librosa.feature.inverse.mel_to_audio(input_gen_cnn_mels, sr=32000, n_fft=1024, hop_length=320, win_length=1024)
    y_gen_pinv = librosa.feature.inverse.mel_to_audio(input_gen_pinv, sr=32000, n_fft=1024, hop_length=320, win_length=1024)
    y = librosa.feature.inverse.mel_to_audio(input, sr=32000, n_fft=1024, hop_length=320, win_length=1024)

    save_path = "./audio_generated/"+filename[:-4]
    if not os.path.exists("./audio_generated/"+filename[:-4]):
        os.makedirs(save_path)
        print(f"Directory '{save_path}' created.")
    else:
        print(f"Directory '{save_path}' already exists.")

    write(save_path + "/" +filename[:-4]+"_generated_from_cnn_logits.wav", 32000, y_gen_cnn_logits)
    write(save_path + "/" +filename[:-4]+"_generated_from_cnn_mels.wav", 32000, y_gen_cnn_mels)
    write(save_path + "/" +filename[:-4]+"_generated_from_pinv.wav", 32000, y_gen_pinv)
    write(save_path + "/" +filename[:-4]+"_generated_from_groundtruth_mel.wav", 32000, y)
    write(save_path + "/" +filename[:-4]+"_original.wav", 32000, x_32k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transform audio into different spectral representations, transcoded or not')

    parser.add_argument('audio_file', type=str,
                        help='Name of the original audio file that should be located in the "audio" folder')
    
    config = parser.parse_args()
    main(config)