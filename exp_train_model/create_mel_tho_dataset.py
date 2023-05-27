#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 10:05:35 2022

@author: user
"""

import csv
import librosa
import numpy as np
import random
from sklearn.model_selection import train_test_split
import yaml
import os
import send2trash
import argparse
import pandas as pd
import os
import sys
# Add the parent directory of the project directory to the module search path
project_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_parent_dir)
import utils.bands_transform as bt

myseed = 71
random.seed(myseed)

def main(config): 

    #used for both full and urban dataset_type 
    devices_keep = ['a']
    #used only for urban dataset type (only outdoors excerpts)
    classes_keep = ['street_pedestrian', 'public_square', 'street_traffic', 'park']
    #used only if split_by_city is set to True
    cities_train = ['london','lyon','milan','paris','stockholm','vienna']
    cities_valid = ['lisbon','prague']
    cities_eval = ['barcelona','helsinki']

    full_audio_dataset_path = config.audio_dataset_path + '/' + config.audio_dataset_name
    split_path = full_audio_dataset_path + "/evaluation_setup"

    #each row of dataset will contain:
    # filename (ex: audio/metro-barcelona-41-1262-s1.wav), label (ex: metro), city (ex: lisbon), device (ex: a)

    train_path = split_path + "/fold1_train.csv"
    eval_path = split_path + "/fold1_evaluate.csv"

    train_df = pd.read_csv(train_path, sep='\t')
    eval_df = pd.read_csv(eval_path, sep='\t')

    dataset_df = pd.concat([train_df, eval_df], ignore_index=True)

    metadata_split = dataset_df.iloc[:, 0].str.split(r'[-.]', expand=True)
    dataset_df[['city', 'device']] = metadata_split.iloc[:, [1, 4]]

    dataset = dataset_df.values

    """
    LABELS DICT
    """

    index_scene = np.unique(np.array(dataset)[:,1])
    index_scene = dict([(str(y),x) for x,y in enumerate(sorted(set(index_scene)))])

    """
    FILTER DATA (optional)
    """

    if config.dataset_type=='urban':
        dataset = urban_dataset(dataset, devices_keep, classes_keep)
        
    if config.dataset_type=='full':
        dataset = full_dataset(dataset, devices_keep)
    
    #tool dataset
    if config.dataset_type=='test':
        dataset = dataset[0:50]

    """
    TRAIN, VALIDATION, AND EVALUATION SPLIT
    """
    if config.split_by_city:
        train_dataset, valid_dataset, eval_dataset = split_dataset(dataset, cities_train, cities_valid, cities_eval)
    else:
        train_dataset, valid_dataset = train_test_split(dataset,test_size=1-config.train_ratio, random_state=myseed)
        valid_dataset, eval_dataset = train_test_split(valid_dataset,test_size=config.eval_ratio/(config.eval_ratio + config.valid_ratio), random_state=myseed)

    print(f'total: {len(dataset)}')
    print(f'train: {len(train_dataset)}')
    print(f'valid: {len(valid_dataset)}')
    print(f'eval: {len(eval_dataset)}')

    """
    GET THIRD-OCTAVE AND MEL SPECTROGRAMS
    """

    if config.mel_template != None:
        #only if you want to test third-octave with Mel freq interpolation (shape (64, 8)), or with Mel time interpolation (shape (29, 101))
        tho_bt = bt.ChangeInterpolationThirdOctaveTransform(32000, 1024, 320, mel_template=config.mel_template, tho_freq=config.tho_freq, tho_time=config.tho_time)
    else:
        #general third-octave transform. Should be used by default.
        tho_bt = bt.ThirdOctaveTransform(32000, config.flen, config.hlen)

    mels_bt_pann = bt.PANNMelsTransform(flen_tho=tho_bt.flen)
    mels_bt_yamnet = bt.YamNetMelsTransform(flen_tho=tho_bt.flen)

            
    if not os.path.exists(config.output_path + '/' + config.dataset_name):
        os.makedirs(config.output_path + '/' + config.dataset_name)
    if not os.path.exists(config.setting_data_path):
        os.makedirs(config.setting_data_path)

    # else:
    #     print(f'WARNING: everything will be deleted in path: {config.output_path / config.dataset_name}')
    #     _delete_everything_in_folder(config.output_path / config.dataset_name)

    all_tho_data_train, all_mels_yamnet_data_train, all_mels_pann_data_train, all_labels_train, all_names_train = extract_spectrograms_from_dataset(train_dataset, full_audio_dataset_path, tho_bt, mels_bt_pann, mels_bt_yamnet, p_label='train')
    all_labels_train_coded = [index_scene[x] for x in all_labels_train]
    np.save(config.output_path + '/' + config.dataset_name + '/' + (config.dataset_name + '_train_third_octave_data'), all_tho_data_train)
    np.save(config.output_path + '/' + config.dataset_name + '/' + (config.dataset_name + '_train_mels_yamnet_data'), all_mels_yamnet_data_train)
    np.save(config.output_path + '/' + config.dataset_name + '/' + (config.dataset_name + '_train_mels_pann_data'), all_mels_pann_data_train)
    np.save(config.output_path + '/' + config.dataset_name + '/' + (config.dataset_name + '_train_scene'), all_labels_train_coded)
    np.save(config.output_path + '/' + config.dataset_name + '/' + (config.dataset_name + '_train_fnames'), all_names_train)
    np.save(config.output_path + '/' + config.dataset_name + '/' + (config.dataset_name + '_train_metadata'), train_dataset)
    del all_tho_data_train, all_mels_yamnet_data_train, all_mels_pann_data_train, all_labels_train, all_labels_train_coded

    all_tho_data_valid, all_mels_yamnet_data_valid, all_mels_pann_data_valid, all_labels_valid, all_names_valid = extract_spectrograms_from_dataset(valid_dataset, full_audio_dataset_path, tho_bt, mels_bt_pann, mels_bt_yamnet, p_label='valid')
    all_labels_valid_coded = [index_scene[x] for x in all_labels_valid]
    np.save(config.output_path + '/' + config.dataset_name + '/' + (config.dataset_name + '_valid_third_octave_data'), all_tho_data_valid)
    np.save(config.output_path + '/' + config.dataset_name + '/' + (config.dataset_name + '_valid_mels_yamnet_data'), all_mels_yamnet_data_valid)
    np.save(config.output_path + '/' + config.dataset_name + '/' + (config.dataset_name + '_valid_mels_pann_data'), all_mels_pann_data_valid)
    np.save(config.output_path + '/' + config.dataset_name + '/' + (config.dataset_name + '_valid_scene'), all_labels_valid_coded)
    np.save(config.output_path + '/' + config.dataset_name + '/' + (config.dataset_name + '_valid_fnames'), all_names_valid)
    np.save(config.output_path + '/' + config.dataset_name + '/' + (config.dataset_name + '_valid_metadata'), valid_dataset)
    del all_tho_data_valid, all_mels_yamnet_data_valid, all_mels_pann_data_valid, all_labels_valid, all_labels_valid_coded

    all_tho_data_eval, all_mels_yamnet_data_eval, all_mels_pann_data_eval, all_labels_eval, all_names_eval = extract_spectrograms_from_dataset(eval_dataset, full_audio_dataset_path, tho_bt, mels_bt_pann, mels_bt_yamnet, p_label='eval')
    all_labels_eval_coded = [index_scene[x] for x in all_labels_eval]
    np.save(config.output_path + '/' + config.dataset_name + '/' + (config.dataset_name + '_eval_third_octave_data'), all_tho_data_eval)
    np.save(config.output_path + '/' + config.dataset_name + '/' + (config.dataset_name + '_eval_mels_yamnet_data'), all_mels_yamnet_data_eval)
    np.save(config.output_path + '/' + config.dataset_name + '/' + (config.dataset_name + '_eval_mels_pann_data'), all_mels_pann_data_eval)
    np.save(config.output_path + '/' + config.dataset_name + '/' + (config.dataset_name + '_eval_scene'), all_labels_eval_coded)
    np.save(config.output_path + '/' + config.dataset_name + '/' + (config.dataset_name + '_eval_fnames'), all_names_eval)
    np.save(config.output_path + '/' + config.dataset_name + '/' + (config.dataset_name + '_eval_metadata'), eval_dataset)
    del all_tho_data_eval, all_mels_yamnet_data_eval, all_mels_pann_data_eval, all_labels_eval, all_labels_eval_coded

    # zip the three name arrays together
    all_names = zip(all_names_train, all_names_valid, all_names_eval)
    csv_file_path = config.output_path + '/' + config.dataset_name + '/' + "train_valid_eval_split.csv"

    # Write the data to the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Train Split", "Valid Split", "Eval Split"])  # Write the header row
        writer.writerows(all_names)  # Write the data rows

    """
    "SAVE SETTINGS"
    """

    data_settings = {
    "index_scene": index_scene,
    "root_dir": 'data',
    "sr" : 32000
    }

    var_config = vars(config)
    for arg_name, arg_value in var_config.items():
        data_settings[arg_name] = arg_value

    with open(config.setting_data_path + '/' + (config.dataset_name + '_settings.yaml'), 'w') as file:
        yaml.dump(data_settings, file)

def urban_dataset(dataset, devices_keep, classes_keep):
    """
    Filters the dataset based on the specified devices and classes.

    Args:
        dataset (list): List of data containing the dataset.
        devices_keep (list): List of devices to keep in the filtered dataset.
        classes_keep (list): List of classes to keep in the filtered dataset.

    Returns:
        list: The filtered dataset based on the specified devices and classes.

    Examples:
        >>> dataset = [...]
        >>> devices_keep = ['a', 's1']
        >>> classes_keep = ['metro', 'airport']
        >>> filtered_dataset = urban_dataset(dataset, devices_keep, classes_keep)
    """

    dataset_f = dataset
    
    if devices_keep:
        dataset_f = [x for x in dataset_f if True in
                    [y == x[-1] for y in devices_keep]]
    if classes_keep:
        dataset_f = [x for x in dataset_f if True in
                    [y == x[1] for y in classes_keep]]
    return(dataset_f)

def full_dataset(dataset, devices_keep):
    """
    Filters the dataset based on the specified devices.

    Args:
        dataset (list): List of data containing the dataset.
        devices_keep (list): List of devices to keep in the filtered dataset.

    Returns:
        list: The filtered dataset based on the specified devices.

    Examples:
        >>> dataset = [...]
        >>> devices_keep = ['a', 's1']
        >>> filtered_dataset = full_dataset(dataset, devices_keep)
    """
    dataset_f = dataset
    
    if devices_keep:
        dataset_f = [x for x in dataset_f if True in
                    [y == x[-1] for y in devices_keep]]
    
    return(dataset_f)

def split_dataset(dataset, cities_train, cities_valid, cities_eval):
    """
    Splits the dataset into train, valid, and eval datasets based on the specified cities.

    Args:
        dataset (list): List of data containing the dataset.
        cities_train (list): List of cities for the train dataset.
        cities_valid (list): List of cities for the valid dataset.
        cities_eval (list): List of cities for the eval dataset.

    Returns:
        tuple: A tuple containing the train, valid, and eval datasets lists based on the specified cities.
    """
    dataset_f = dataset
    
    if cities_train:
        train_dataset = [x for x in dataset_f if True in
                    [y == x[2] for y in cities_train]]
    
    if cities_valid:
        valid_dataset = [x for x in dataset_f if True in
                    [y == x[2] for y in cities_valid]] 
    
    if cities_eval:
        eval_dataset = [x for x in dataset_f if True in
                    [y == x[2] for y in cities_eval]] 
        
    return(train_dataset, valid_dataset, eval_dataset)
    
def extract_spectrograms_from_dataset(dataset, full_audio_dataset_path, tho_bt, mels_bt_pann, mels_bt_yamnet, p_label='default'):
    """
    Extracts third-octave and mel spectrograms from the audio files in the dataset.

    Args:
        dataset (list): List of data containing audio file names and labels (ex: [['audio/metro_station-prague-1130-42569-a.wav_1', 'metro'], ...]).
        full_audio_dataset_path (str): Path to the full audio dataset.
        tho_bt (object): Third-octave band transformer object.
        mels_bt_pann (object): Mel bands transformer object for PANN.
        mels_bt_yamnet (object): Mel bands transformer object for YamNet.
        p_label (str, optional): Label for the dataset (default is 'default'). Just used for display.

    Returns:
        tuple: A tuple containing the extracted third-octave data, YamNet mel spectrograms data, PANN mel spectrograms data,
        labels, and names of the processed audio chunks.
    """

    all_tho_data = []
    all_mels_pann_data = []
    all_mels_yamnet_data = []
    all_labels = []
    all_names = []
    n_file = 0

    for data in dataset:
        data_name = full_audio_dataset_path + '/' + data[0]

        audio_file = librosa.load(data_name, sr=32000)[0]
        #MT: not sure if normalizing is necessary, doing it again just in case
        audio_file = librosa.util.normalize(audio_file) 
        
        #split each 10s audio excerpt into 1s audio chunks
        n_frames_in_chunk = int(len(audio_file)/10)
        audio_file_split = list(chunks(audio_file, n_frames_in_chunk))
        
        #for each chunkv, calculate third-octave and mel spectrograms
        for idx, audio_file_chunk in enumerate(audio_file_split):
            # third-octave spectrogram extraction
            x_tho = tho_bt.wave_to_third_octave(np.array(audio_file_chunk), zeropad=True)
            all_tho_data.append(x_tho.T)

            # mel spectrogram extraction
            x_mels_pann = mels_bt_pann.wave_to_mels(np.array(audio_file_chunk))
            all_mels_pann_data.append(x_mels_pann.T)

            #store label and file name of the chunk, with each chunk having its index at the end of the filanme (ex: audio/metro_station-prague-1130-42569-a.wav_6)
            all_labels.append(data[1])
            all_names.append(data[0]+'_'+str(idx))
        
        #MT: YamNet is sampled at 16kHz
        audio_file = librosa.load(data_name, sr=16000)[0]
        #MT: not sure if normalizing is necessary, doing it again just in case
        audio_file = librosa.util.normalize(audio_file) 
        
        n_frames_in_chunk = int(len(audio_file)/10)
        audio_file_split = list(chunks(audio_file, n_frames_in_chunk))

        for audio_file_chunk in audio_file_split:
            # mel spectrogram extraction
            x_mels_yamnet = mels_bt_yamnet.wave_to_mels(np.array(audio_file_chunk))
            all_mels_yamnet_data.append(x_mels_yamnet.T)
            
        n_file += 1
        print('\r' + f'{n_file} files have been processed in {p_label} dataset',end=' ')

    all_tho_data = np.array(all_tho_data)
    all_mels_yamnet_data = np.array(all_mels_yamnet_data)
    all_mels_pann_data = np.array(all_mels_pann_data)
    all_labels = np.array(all_labels)
    all_names = np.array(all_names)

    print(f'\n {p_label} done')
    return(all_tho_data, all_mels_yamnet_data, all_mels_pann_data, all_labels, all_names)

def chunks(lst, n):
    """
    Yield successive n-sized chunks from a list.

    Args:
        lst (list): The input list.
        n (int): The size of each chunk.

    Yields:
        list: A chunk of size n from the input list.

    Examples:
        >>> lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> for chunk in chunks(lst, 3):
        ...     print(chunk)
        [1, 2, 3]
        [4, 5, 6]
        [7, 8, 9]
        [10]
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def _delete_everything_in_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                send2trash(file_path)
                #shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate 1s Mels and Third-Octave spectrograms')

    parser.add_argument('--dataset_name', type=str, default='Dcase-Task1-full',
                        help='The name of the dataset')
    parser.add_argument('--audio_dataset_path', type=str, default='.',
                        help='The name of the dataset')
    parser.add_argument('--audio_dataset_name', type=str, default='TAU-urban-acoustic-scenes-2020-mobile-development',
                        help='The name of the dataset')
    parser.add_argument('--output_path', type=str, default='./spectral_transcoder_data/data',
                        help='The name of the dataset')
    parser.add_argument('--setting_data_path', type=str, default='./spectral_transcoder_data/data_settings',
                        help='The name of the dataset')
    
    parser.add_argument('--train_ratio', type=float, default=0.75,
                        help='The ratio of data to use for training')
    parser.add_argument('--valid_ratio', type=float, default=0.15,
                        help='The ratio of data to use for validation')
    parser.add_argument('--eval_ratio', type=float, default=0.15,
                        help='The ratio of data to use for evaluation')

    parser.add_argument('--dataset_type', type=str, default='full',
                        help='Whether to filter the data (full, urban, test)')
    parser.add_argument('--devices_keep', type=bool, default=['a'],
                        help='Which devices to keep (a,b,c etc...)')
    parser.add_argument('--split_by_city', type=bool, default=False,
                        help='Whether to do the train/test/validation split by city or not')
    parser.add_argument('--tho_freq', type=bool, default=True,
                        help='The frequency to keep')
    parser.add_argument('--tho_time', type=float, default=True,
                        help='The time to keep')
    parser.add_argument('--mel_template', type=str, default=None,
                        help='The MEL template to use, in case mel_freq or mel_time (tho_freq=False or mel_freq=False)')
    parser.add_argument('--flen', type=float, default=4096,
                        help='Window length for the stft of the third-octave spectrogram')
    parser.add_argument('--hlen', type=float, default=4000,
                        help='Hop length for the stft of the third-octave spectrogram')
    
    config = parser.parse_args()
    main(config)