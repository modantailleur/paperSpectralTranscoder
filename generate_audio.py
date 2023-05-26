
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
    #PROJECT_DATA_PATH = Path("/home/user/Documents/from_pc_lagrange/doce_experiment_08032023/")
    #PROJECT_DATA_PATH = Path("/media/user/MT-SSD-NEW/0-PROJETS_INFO/Thèse/exp_01042023/BACK-2-ThirdOToMel-data/doce_experiments/exp_with_pann_resnet38/")
    MODEL_PATH = "./reference_models"

    #filename = "street_pedestrian-paris-152-4607-a.wav"
    #filename = "street_traffic-stockholm-174-5353-a.wav"
    #filename = "park-vienna-105-2989-a.wav"
    #filename = "park-lisbon-1104-43701-a.wav"
    #filename = "park-london-96-2684-a.wav"
    #filename = "park-vienna-104-2944-a.wav"
    #filename = "street_pedestrian-prague-1051-43543-a.wav"
    #filename = "CTED_Singer16_ClearVoice_High_lyrics.wav"
    #filename = "dog_bark.wav"
    #filename = "birds.wav"
    #filename = "voice_male.wav"
    #filename = "voice_female.wav"
    #filename = "voice_countdown.wav"
    filename = config.audio_file

    #filename = "62b89b3078257cc153cd56cc-5.wav"
    #filename = "165192__tagirov__istanbul-dervish-cafe-music.wav"
    #filename = "219447__tom_woysky__crowd-applause-at-the-end-of-concert.wav"
    #filename = "365916__inspectorj__train-passing-close-right-to-left-b.wav"
    #filename = "81056__inplano__retro-airplane.wav"
    #filename = "352514__inspectorj__ambience-night-wildlife-a.wav"
    #filename = "515369__tripjazz__frogs.wav"
    #filename = "536219__ambientsoundapp__bicycles-passing.wav"
    #model_name = 'model_classifier=PANN+dataset=urban+dilation=1+epoch=100+kernel_size=5+learning_rate=-3+nb_channels=64+nb_layers=5+prop_logit=100+step=compute+transcoder=cnn_pinv'
    model_name = 'classifier=PANN+dataset=full+dilation=1+epoch=200+kernel_size=5+learning_rate=-3+nb_channels=64+nb_layers=5+prop_logit=100+step=train+transcoder=cnn_pinv+ts=1_model'
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
    #x_32k = x_32k[0:32000]

    # tho_tr = bt.ThirdOctaveTransform(sr=32000, flen=4096, hlen=4000)

    # mels_tr = bt.PANNMelsTransform()

    transcoder_deep_pann = ThirdOctaveToMelTranscoder(transcoder, model_name, MODEL_PATH, device=device)
    transcoder_pinv_pann = ThirdOctaveToMelTranscoderPinv(MODEL_PATH, model_name, device, classifier="PANN")

    #calculate third octave spectrum
    # x_tho = tho_tr.wave_to_third_octave(x_32k)
    # x_tho = torch.from_numpy(x_tho.T)
    # x_tho = x_tho.unsqueeze(0)
    # x_tho = x_tho.type(dtype)

    x_32k = librosa.load(full_filename, sr=fs)[0]
    x_32k = librosa.util.normalize(x_32k)

    x_mels_pann_cnn = transcoder_deep_pann.transcode_from_wav_entire_file(x_32k)
    x_logit_pann_cnn = transcoder_deep_pann.mels_to_logit_entire_file(x_mels_pann_cnn, slice=False)
    x_mels_pann_gt = transcoder_deep_pann.mels_tr.wave_to_mels(x_32k)
    x_logit_pann_gt = transcoder_deep_pann.mels_to_logit_entire_file(x_mels_pann_gt, slice=False)
    x_mels_pann_pinv = transcoder_pinv_pann.transcode_from_wav(x_32k)



    # #calculate predicted mels
    # x_mels_inf_deep, x_logit_deep = transcoder_deep.transcode_from_wav(x_32k)
    # x_mels_inf_pinv = transcoder_pinv.transcode_from_wav(x_32k)
    # x_mels_gt = transcoder_deep.mels_tr.wave_to_mels(x_32k)
    # x_logit_gt = transcoder_deep.mels_to_logit(x_mels_gt)

    print('\n XXXXXXXXX ORIGINAL PANN CLASSIFIER (MEL INPUT) XXXXXXXXXXX')
    labels = sort_labels_by_score(np.mean(x_logit_pann_gt, axis=1), transcoder_deep_pann.classif_inference.labels_str)[1][:10]
    scores = sort_labels_by_score(np.mean(x_logit_pann_gt, axis=1), transcoder_deep_pann.classif_inference.labels_str)[0][:10]
    for k in range(len(labels)):
        print(f'{labels[k]} : {round(float(scores[k]), 2)}')

    print('\n XXXXXXXXXXXX TRANSCODED PANN CLASSIFIER (THIRD-OCTAVE INPUT) XXXXXXXXXXXX')
    labels = sort_labels_by_score(np.mean(x_logit_pann_cnn, axis=1), transcoder_deep_pann.classif_inference.labels_str)[1][:10]
    scores = sort_labels_by_score(np.mean(x_logit_pann_cnn, axis=1), transcoder_deep_pann.classif_inference.labels_str)[0][:10]
    for k in range(len(labels)):
        print(f'{labels[k]} : {round(float(scores[k]), 2)}')

    # reactivate if we want to save as wav the result
    #input must be of size (n_mel, n)
    input_gen_deep = x_mels_pann_cnn
    input_gen_pinv = x_mels_pann_pinv
    input = x_mels_pann_gt

    y_gen_deep = librosa.feature.inverse.mel_to_audio(input_gen_deep, sr=32000, n_fft=1024, hop_length=320, win_length=1024)
    y_gen_pinv = librosa.feature.inverse.mel_to_audio(input_gen_pinv, sr=32000, n_fft=1024, hop_length=320, win_length=1024)
    y = librosa.feature.inverse.mel_to_audio(input, sr=32000, n_fft=1024, hop_length=320, win_length=1024)

    save_path = "./audio_generated/"+filename[:-4]
    if not os.path.exists("./audio_generated/"+filename[:-4]):
        os.makedirs(save_path)
        print(f"Directory '{save_path}' created.")
    else:
        print(f"Directory '{save_path}' already exists.")

    write(save_path + "/" +filename[:-4]+"_generated_from_transcoder.wav", 32000, y_gen_deep)
    write(save_path + "/" +filename[:-4]+"_generated_from_pinv.wav", 32000, y_gen_pinv)
    write(save_path + "/" +filename[:-4]+"_generated_from_groundtruth_mel.wav", 32000, y)
    write(save_path + "/" +filename[:-4]+"_original.wav", 32000, x_32k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transform audio into different spectral representations, transcoded or not')

    parser.add_argument('audio_file', type=str,
                        help='Name of the original audio file that should be located in the "audio" folder')
    
    config = parser.parse_args()
    main(config)