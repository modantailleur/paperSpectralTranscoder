#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 14:36:19 2022

@author: user
"""

import os
import torch
import torch.utils.data
import numpy as np
from pathlib import Path

class MelDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for training the models that are trained on Mels. This dataset
    class doesn't support training on logits (teacher-student approach)

    Args:
        setting_data (dict): A dictionary containing the dataset settings.
        n_tho_frames_per_file (int): The length of temporal frames.
        subset (str): The subset of the dataset to load (e.g., 'train', 'valid', 'test').
        classifier (str): The classifier type ('YamNet' or 'PANN').
        project_data_path (Path): The path to the project data directory.

    Attributes:

        --> Attributes from the data yaml file

        dataset_name (str): The name of the dataset.
        dataset_path (Path): The path to the dataset directory.
        full_path (Path): The full path to the dataset, concatenation of dataset_name and dataset_path
        sr (int): The sample rate of the audio.
        flen (int): The frame length of the third-octave data.
        hlen (int): The hop length of the third-octave data.
        train_ratio (float): The ratio of training data.
        valid_ratio (float): The ratio of validation data.
        eval_ratio (float): The ratio of evaluation data.
        index_scene (int): The list of scene indices (correspondance between scene name and number)
        subset (str): The subset of the dataset (train, valid or eval)

        --> Other attributes

        n_tho_frames_per_file (int): The length of temporal frames.
        classifier (str): The classifier type.
        mels_root (str): The root name for Mels data based on the classifier. This is the name used to save the create the Mel
            and third-octave dataset with create_mel_tho_dataset.py
        data_tho (numpy.ndarray): The third octave data.
        data_mel (numpy.ndarray): The Mels data.
        metadata (numpy.ndarray): The metadata.
        labels (numpy.ndarray): The labels.
        fnames (numpy.ndarray): The file names.
        n_tho_frames (int): The total number of frames contained in the dataset.
        n_tho (int): The number of third octave bins.
        n_mels (int): The number of Mels bins.
        n_mel_frames_per_file (int): The length of temporal frames for Mels data.

    Methods:
        __getitem__(self, idx): Retrieves a data sample and its corresponding index.
        __len__(self): Returns the number of frames in the dataset.
    """
    def __init__(self, setting_data, n_tho_frames_per_file=8, subset='train', classifier='YamNet', project_data_path=Path().absolute()):

        self.dataset_name = setting_data['dataset_name']
        self.dataset_path = os.path.join(setting_data['root_dir'], self.dataset_name)
        self.dataset_path = project_data_path / self.dataset_path
        self.sr = setting_data['sr']
        self.flen = setting_data['flen']
        self.hlen = setting_data['hlen']
        self.train_ratio = setting_data['train_ratio']
        self.valid_ratio = setting_data['valid_ratio']
        self.eval_ratio = setting_data['eval_ratio']
        self.index_scene = setting_data['index_scene']
        self.subset = subset
        self.n_tho_frames_per_file = n_tho_frames_per_file

        #mmap used to memory-map the file. It is stored in disk, but 
        #small fragments of the file can be accessed without reading the
        #entire file into memory. I removed mmap on data_tho and 
        #data_mel because of the warning "UserWarning: The given NumPy array 
        #is not writable"
        self.full_path = self.dataset_path / self.dataset_name
        
        self.classifier = classifier
        if classifier == "PANN":
            self.mels_root = 'mels_pann'
        if classifier == 'YamNet':
            self.mels_root = 'mels_yamnet'
            
        self.data_tho = np.load(str(self.full_path) +'_'+self.subset+'_third_octave_data.npy', mmap_mode='r')
        self.data_mel = np.load(str(self.full_path) +'_'+self.subset+'_'+self.mels_root+'_data.npy', mmap_mode='r')
        #self.metadata = np.load(str(self.full_path) +'_'+self.subset+'_metadata.npy', mmap_mode='r')
        #self.scene = np.load(str(self.full_path) +'_'+self.subset+'_scene.npy', mmap_mode='r')
        self.fnames = np.load(str(self.full_path) +'_'+self.subset+'_fnames.npy', mmap_mode='r')
        
        self.n_tho_frames = np.shape(self.data_tho)[0]
        self.n_tho = np.shape(self.data_tho)[2]
        self.n_mels = np.shape(self.data_mel)[2]
        self.n_mel_frames_per_file = np.shape(self.data_mel)[1]
            
    def __getitem__(self, idx):
        
        input_tho = torch.from_numpy(np.copy(self.data_tho[idx]))
        input_mel = torch.from_numpy(np.copy(self.data_mel[idx]))
        input_fname = self.fnames[idx]
        
        return (idx, input_tho , input_mel, input_fname)

    def __len__(self):
        return self.n_tho_frames

'''
Used for training Teacher Student models
'''
class MelLogitDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for training the models that are trained on logits and on mels.

    Args:
        setting_data (dict): A dictionary containing the dataset settings.
        n_tho_frames_per_file (int): The length of temporal frames.
        subset (str): The subset of the dataset to load (e.g., 'train', 'valid', 'test').
        classifier (str): The classifier type ('YamNet' or 'PANN').
        project_data_path (Path): The path to the project data directory.

    Attributes:

        --> Attributes from the data yaml file

        dataset_name (str): The name of the dataset.
        dataset_path (Path): The path to the dataset directory.
        full_path (Path): The full path to the dataset, concatenation of dataset_name and dataset_path
        sr (int): The sample rate of the audio.
        flen (int): The frame length of the third-octave data.
        hlen (int): The hop length of the third-octave data.
        train_ratio (float): The ratio of training data.
        valid_ratio (float): The ratio of validation data.
        eval_ratio (float): The ratio of evaluation data.
        index_scene (int): The list of scene indices (correspondance between scene name and number)
        subset (str): The subset of the dataset (train, valid or eval)

        --> Other attributes

        n_tho_frames_per_file (int): The length of temporal frames.
        classifier (str): The classifier type.
        mels_root (str): The root name for Mels data based on the classifier. This is the name used to save the create the Mel
            and third-octave dataset with create_mel_tho_dataset.py
        data_tho (numpy.ndarray): The third octave data.
        data_mel (numpy.ndarray): The Mels data.
        metadata (numpy.ndarray): The metadata.
        labels (numpy.ndarray): The labels.
        fnames (numpy.ndarray): The file names.
        n_tho_frames (int): The total number of frames contained in the dataset.
        n_tho (int): The number of third octave bins.
        n_mels (int): The number of Mels bins.
        n_mel_frames_per_file (int): The length of temporal frames for Mels data.

    Methods:
        __getitem__(self, idx): Retrieves a data sample and its corresponding index.
        __len__(self): Returns the number of frames in the dataset.
    """
    def __init__(self, setting_data, outputs_oracle_path, n_tho_frames_per_file=8, subset='train',
                  classifier='YamNet', project_data_path=Path().absolute()):
        
        #mels and third octaves part
        self.dataset_name = setting_data['dataset_name']
        self.dataset_path = os.path.join(setting_data['root_dir'], self.dataset_name)
        self.dataset_path = project_data_path / self.dataset_path
        self.sr = setting_data['sr']
        self.flen = setting_data['flen']
        self.hlen = setting_data['hlen']
        self.train_ratio = setting_data['train_ratio']
        self.valid_ratio = setting_data['valid_ratio']
        self.eval_ratio = setting_data['eval_ratio']
        self.index_scene = setting_data['index_scene']
        self.subset = subset
        self.n_tho_frames_per_file = n_tho_frames_per_file
        
        #predictions part
        #self.dataset_name = setting_data['data']['dataset_name']
        self.batch_type_name = classifier
        self.dummy_dataset = MelDataset(setting_data, subset='eval', classifier=classifier, project_data_path=project_data_path)

        def totuple(a):
            try:
                return tuple(totuple(i) for i in a)
            except TypeError:
                return a
            
        shape_predictions = totuple(np.load(outputs_oracle_path['logits']+'_'+self.subset+'_shape.npy'))
        shape_predictions_filtered = totuple(np.load(outputs_oracle_path['logits_tvb']+'_'+self.subset+'_shape.npy'))

        #mmap used to memory-map the file. It is stored in disk, but 
        #small fragments of the file can be accessed without reading the
        #entire file into memory. I removed mmap on data_tho and 
        #data_mel because of the warning "UserWarning: The given NumPy array 
        #is not writable"
        self.oracle_logits = np.memmap(outputs_oracle_path['logits']+'_'+self.subset+'.dat', dtype=np.float64, mode ='r',shape=shape_predictions)
        self.oracle_logits_tvb = np.memmap(outputs_oracle_path['logits_tvb']+'_'+self.subset+'.dat', dtype=np.float64, mode ='r', shape=shape_predictions_filtered)
                
        self.n_files = np.shape(self.oracle_logits)[0]
        self.full_path = self.dataset_path / self.dataset_name
        
        self.mels_type = classifier
        if classifier == "PANN":
            self.mels_root = 'mels_pann'
        if classifier == 'YamNet':
            self.mels_root = 'mels_yamnet'
            
        self.data_tho = np.load(str(self.full_path) +'_'+self.subset+'_third_octave_data.npy', mmap_mode='r')
        self.data_mel = np.load(str(self.full_path) +'_'+self.subset+'_'+self.mels_root+'_data.npy', mmap_mode='r')
        #self.metadata = np.load(str(self.full_path) +'_'+self.subset+'_metadata.npy', mmap_mode='r')
        #self.scene = np.load(str(self.full_path) +'_'+self.subset+'_scene.npy', mmap_mode='r')
        self.fnames = np.load(str(self.full_path) +'_'+self.subset+'_fnames.npy', mmap_mode='r')
        
        self.n_tho_frames = np.shape(self.data_tho)[0]
        self.n_tho = np.shape(self.data_tho)[2]
        self.n_mels = np.shape(self.data_mel)[2]
        self.n_mel_frames_per_file = np.shape(self.data_mel)[1]
        
    def __getitem__(self, idx):
        
        input_tho = torch.from_numpy(np.copy(self.data_tho[idx]))
        input_mel = torch.from_numpy(np.copy(self.data_mel[idx]))
        input_fname = self.fnames[idx]
        
        oracle_logit = torch.from_numpy(np.copy(self.oracle_logits[idx]))
        oracle_logit_tvb = torch.from_numpy(np.copy(self.oracle_logits_tvb[idx]))
        
        return (idx, input_tho , input_mel, input_fname, oracle_logit, oracle_logit_tvb)

    def __len__(self):
        return self.n_tho_frames

class OutputsDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for loading output data generated by a model. This is only used for metrics calculation,
    and returns outputs data for exactly 10s audio excerpts. 

    Args:
        setting_data (dict): A dictionary containing setting data for the dataset.
        outputs_path (str): Path to the directory containing the model's output data.
        outputs_oracle_path (str): Path to the directory containing the oracle output data.
        no_mels (bool, optional): If True, excludes mel-spectrogram data from the dataset. Defaults to False.
        classifier (str, optional): The classifier type. Defaults to 'YamNet'.
        project_data_path (str, optional): The absolute path to the project data directory. Defaults to the current working directory.

    Attributes:
        dataset_name (str): The name of the dataset.
        batch_type_name (str): The name of the classifier.
        dummy_dataset (MelDataset): An instance of the MelDataset class used to access evaluation data.
        no_mels (bool): True if mel-spectrogram data is excluded from the dataset, False otherwise.
        oracle_mels (np.memmap or None): Memory-mapped array of oracle mel-spectrogram data. None if no_mels is True.
        model_mels (np.memmap or None): Memory-mapped array of model mel-spectrogram data. None if no_mels is True.
        oracle_logits (np.memmap): Memory-mapped array of oracle logit data.
        model_logits (np.memmap): Memory-mapped array of model logit data.
        n_files (int): The number of files in the dataset.

    Note:
        The dataset expects the model's output data to be stored as memory-mapped numpy arrays.

    """
    def __init__(self, setting_data,
                  outputs_path, outputs_oracle_path, no_mels=False,
                  classifier='YamNet', project_data_path=Path().absolute()):
        
        #mmap used to memory-map the file. It is stored in disk, but 
        #small fragments of the file can be accessed without reading the
        #entire file into memory. I removed mmap on data_tho and 
        #data_mel because of the warning "UserWarning: The given NumPy array 
        #is not writable"
        self.dataset_name = setting_data['dataset_name']
        self.batch_type_name = classifier
        self.dummy_dataset = MelDataset(setting_data, subset='eval', classifier=classifier, project_data_path=project_data_path)
        self.no_mels = no_mels

        def totuple(a):
            try:
                return tuple(totuple(i) for i in a)
            except TypeError:
                return a
            
        shape_predictions = totuple(np.load(outputs_oracle_path['logits']+'_shape.npy'))
        #shape_predictions_filtered = totuple(np.load(outputs_oracle_path['logits_tvb']+'_shape.npy'))

        if self.no_mels:
            self.oracle_mels = None
            self.models_mels = None
        else:
            shape_mels = totuple(np.load(outputs_oracle_path['mels']+'_shape.npy'))
            
            self.oracle_mels = np.memmap(outputs_oracle_path['mels']+'.dat',  dtype=np.float64, mode ='r', shape=shape_mels)
            self.model_mels = np.memmap(outputs_path['mels']+'.dat', mode ='r',dtype=np.float64, shape=shape_mels)
        
        shape_logits = totuple(np.load(outputs_oracle_path['logits']+'_shape.npy'))
        self.oracle_logits = np.memmap(outputs_oracle_path['logits']+'.dat', dtype=np.float64, mode ='r',shape=shape_logits)
        #self.oracle_logits_tvb = np.memmap(outputs_oracle_path['logits_tvb']+'.dat', dtype=np.float64, mode ='r', shape=shape_predictions_filtered)
        
        self.model_logits = np.memmap(outputs_path['logits']+'.dat', mode ='r', dtype=np.float64, shape=shape_logits)
        #self.model_logits_tvb = np.memmap(outputs_path['logits_tvb']+'.dat', dtype=np.float64, mode ='r', shape=shape_predictions_filtered)
        
        self.n_files = np.shape(self.oracle_logits)[0]
                    
    def __getitem__(self, idx):
        
        if self.no_mels:
            oracle_mel = torch.Tensor([])
            model_mel = torch.Tensor([])
        else:
            oracle_mel = torch.from_numpy(np.copy(self.oracle_mels[idx*10: idx*10+10]))
            model_mel = torch.from_numpy(np.copy(self.model_mels[idx*10: idx*10+10]))
        
        oracle_logit = torch.from_numpy(np.copy(self.oracle_logits[idx*10: idx*10+10]))
        #oracle_logit_tvb = torch.from_numpy(np.copy(self.oracle_logits_tvb[idx*10: idx*10+10]))
        
        model_logit = torch.from_numpy(np.copy(self.model_logits[idx*10: idx*10+10]))
        #model_logit_tvb = torch.from_numpy(np.copy(self.model_logits_tvb[idx*10: idx*10+10]))
        
        return (oracle_mel, model_mel, oracle_logit, model_logit)

    def __len__(self):
        return int(self.n_files/10)


'''
Evaluation for 1s
'''
# class MelOutputsDataset(torch.utils.data.Dataset):
#     def __init__(self, setting_data,
#                   outputs_path, outputs_oracle_path,
#                   mels_type='YamNet', project_data_path=Path().absolute()):
        
#         #mmap used to memory-map the file. It is stored in disk, but 
#         #small fragments of the file can be accessed without reading the
#         #entire file into memory. I removed mmap on data_tho and 
#         #data_mel because of the warning "UserWarning: The given NumPy array 
#         #is not writable"
#         self.dataset_name = setting_data['data']['dataset_name']
#         self.batch_type_name = mels_type
#         self.dummy_dataset = MelDataset(setting_data, subset='eval', mels_type=mels_type, project_data_path=project_data_path)
#         def totuple(a):
#             try:
#                 return tuple(totuple(i) for i in a)
#             except TypeError:
#                 return a
            
#         shape_mels = totuple(np.load(outputs_oracle_path['mels']+'_shape.npy'))
        
#         self.oracle_mels = np.memmap(outputs_oracle_path['mels']+'.dat',  dtype=np.float64, mode ='r', shape=shape_mels)
        
#         self.model_mels = np.memmap(outputs_path['mels']+'.dat', mode ='r',dtype=np.float64, shape=shape_mels)
                
#         self.n_files = np.shape(self.oracle_mels)[0]
        
            
#     def __getitem__(self, idx):

#         oracle_mel = torch.from_numpy(np.copy(self.oracle_mels[idx]))
        
#         model_mel = torch.from_numpy(np.copy(self.model_mels[idx]))

#         return (oracle_mel, model_mel)

#     def __len__(self):
#         return self.n_files

'''
OLD
'''

# class OutputsDataset(torch.utils.data.Dataset):
#     def __init__(self, setting_data,
#                   outputs_path, outputs_oracle_path,
#                   mels_type='YamNet', project_data_path=Path().absolute()):
        
#         #mmap used to memory-map the file. It is stored in disk, but 
#         #small fragments of the file can be accessed without reading the
#         #entire file into memory. I removed mmap on data_tho and 
#         #data_mel because of the warning "UserWarning: The given NumPy array 
#         #is not writable"
#         self.dataset_name = setting_data['data']['dataset_name']
#         self.batch_type_name = mels_type
#         self.dummy_dataset = MelDataset(setting_data, subset='eval', mels_type=mels_type, project_data_path=project_data_path)
#         # self.model_predictions_name = self.batch_type_name + '_' + self.model_name + '_' + self.dataset_name
#         # self.model_predictions_path = project_data_path / 'outputs' / 'predictions' / model_name / self.model_predictions_name
        
#         # self.model_predictions_filtered_name = self.batch_type_name + '_' + self.model_name + '_' + self.dataset_name
#         # self.model_predictions_filtered_path = project_data_path / 'outputs' / 'predictions_filtered' / model_name / self.model_predictions_name
        
#         # self.model_mels_name =  self.batch_type_name + '_' + self.model_name + '_' + self.dataset_name
#         # self.model_mels_path = project_data_path / 'outputs' / 'mels' / model_name / self.model_mels_name
        
#         # self.oracle_predictions_path =  project_data_path / 'outputs' / 'predictions' / 'oracle' / (self.batch_type_name + '_oracle_' + self.dataset_name)
#         # self.oracle_predictions_name = self.batch_type_name + '_oracle_' + self.dataset_name
        
#         # self.oracle_predictions_filtered_path =  project_data_path / 'outputs' / 'predictions_filtered' / 'oracle' / (self.batch_type_name + '_oracle_' + self.dataset_name)
#         # self.oracle_predictions_filtered_name = self.batch_type_name + '_oracle_' + self.dataset_name
        
#         # self.oracle_mels_path =  project_data_path / 'outputs' / 'mels' / 'oracle' / (self.batch_type_name + '_oracle_' + self.dataset_name)
#         # self.oracle_mels_name = self.batch_type_name + '_oracle_' + self.dataset_name

#         # self.full_predictions_path = self.predictions_path / self.predictions_name
#         # self.full_mels_path = self.mels_path / self.mels_name
#         def totuple(a):
#             try:
#                 return tuple(totuple(i) for i in a)
#             except TypeError:
#                 return a
            
#         shape_mels = totuple(np.load(outputs_oracle_path['mels']+'_shape.npy'))
#         shape_predictions = totuple(np.load(outputs_oracle_path['predictions']+'_shape.npy'))
#         shape_predictions_filtered = totuple(np.load(outputs_oracle_path['predictions_filtered']+'_shape.npy'))
        
#         self.oracle_predictions = np.memmap(outputs_oracle_path['predictions']+'.dat', dtype=np.float64, mode ='r',shape=shape_predictions)
#         self.oracle_predictions_filtered = np.memmap(outputs_oracle_path['predictions_filtered']+'.dat', dtype=np.float64, mode ='r', shape=shape_predictions_filtered)
#         self.oracle_mels = np.memmap(outputs_oracle_path['mels']+'.dat',  dtype=np.float64, mode ='r', shape=shape_mels)
        
#         self.model_predictions = np.memmap(outputs_path['predictions']+'.dat', mode ='r', dtype=np.float64, shape=shape_predictions)
#         self.model_predictions_filtered = np.memmap(outputs_path['predictions_filtered']+'.dat', dtype=np.float64, mode ='r', shape=shape_predictions_filtered)
#         self.model_mels = np.memmap(outputs_path['mels']+'.dat', mode ='r',dtype=np.float64, shape=shape_mels)
        
#         # self.oracle_predictions = torch.from_numpy(self.oracle_predictions)
#         # self.oracle_predictions_filtered = torch.from_numpy(self.oracle_predictions_filtered)
#         # self.oracle_mels = torch.from_numpy(self.oracle_mels)
        
#         # self.model_predictions = torch.from_numpy(self.model_predictions)
#         # self.model_predictions_filtered = torch.from_numpy(self.model_predictions_filtered)
#         # self.model_mels = torch.from_numpy(self.model_mels)
                
#         self.n_files = np.shape(self.oracle_predictions)[0]
        
            
#     def __getitem__(self, idx):

#         oracle_prediction = torch.from_numpy(np.copy(self.oracle_predictions[idx]))
#         oracle_prediction_filtered = torch.from_numpy(np.copy(self.oracle_predictions_filtered[idx]))
#         oracle_mel = torch.from_numpy(np.copy(self.oracle_mels[idx]))
        
#         model_prediction = torch.from_numpy(np.copy(self.model_predictions[idx]))
#         model_prediction_filtered = torch.from_numpy(np.copy(self.model_predictions_filtered[idx]))
#         model_mel = torch.from_numpy(np.copy(self.model_mels[idx]))

#         return (oracle_prediction, oracle_prediction_filtered, oracle_mel, model_prediction, model_prediction_filtered, model_mel)

#     def __len__(self):
#         return self.n_files

'''
OLD
'''

# class MelDataset(torch.utils.data.Dataset):
#     def __init__(self, setting_data, n_tho_frames_per_file=8, subset='train', mels_type='YamNet', oracle_predictions=None):

#         self.audio_dir = setting_data['data']['audio_dir']
#         self.dataset_name = setting_data['data']['dataset_name']

#         self.dataset_path = os.path.join(setting_data['data']['root_dir'], self.dataset_name)
#         self.dataset_path = Path().absolute() / self.dataset_path
        
#         self.sr = setting_data['data']['sr']
#         self.pad_examples = setting_data['data']['pad_examples']
#         self.flen = setting_data['data']['frame_length']
#         self.hlen = setting_data['data']['hop_length']
#         self.train_ratio = setting_data['data']['train_ratio']
#         self.valid_ratio = setting_data['data']['valid_ratio']
#         self.eval_ratio = setting_data['data']['eval_ratio']
#         self.dbtype = setting_data['data']['dbtype']
#         self.n_mels = setting_data['data']['n_mels']
#         self.labels_dict = setting_data['data']['labels_dict']
#         self.subset = subset
#         self.n_tho_frames_per_file = n_tho_frames_per_file
        
#         #MT: np array that contains the oracle predictions
#         self.oracle_predictions = oracle_predictions
#         self.nb_data_processed = 0
#         self.len_cur_chunk = 0

#         #mmap used to memory-map the file. It is stored in disk, but 
#         #small fragments of the file can be accessed without reading the
#         #entire file into memory. I removed mmap on data_tho and 
#         #data_mel because of the warning "UserWarning: The given NumPy array 
#         #is not writable"
#         self.full_path = self.dataset_path / self.dataset_name
#         self.cur_chunk = 0
#         self.n_chunks = self._count_chunks()
        
#         self.mels_type = mels_type
#         if mels_type == "PANN":
#             self.mels_root = 'mels_pann'
#         if mels_type == 'YamNet':
#             self.mels_root = 'mels_yamnet'
            
#         self.load_next_chunk(init=True)
        
#     def load_next_chunk(self, init=False):

#         if init:
#             self.cur_chunk=0
#         else:
#             self.cur_chunk+=1  
            
#         if self.cur_chunk >= self.n_chunks:
#             #print('no more chunks')
#             return
        
#         self.nb_data_processed = self.nb_data_processed + self.len_cur_chunk
        
#         self.data_tho = np.load(str(self.full_path) +'_'+self.subset+'_third_octave_data_chunk'+str(self.cur_chunk)+'.npy', mmap_mode='r+')
#         self.data_mel = np.load(str(self.full_path) +'_'+self.subset+'_'+self.mels_root+'_data_chunk'+str(self.cur_chunk)+'.npy', mmap_mode='r+')
#         self.metadata = np.load(str(self.full_path) +'_'+self.subset+'_metadata_chunk'+str(self.cur_chunk)+'.npy', mmap_mode='r+')
#         self.labels = np.load(str(self.full_path) +'_'+self.subset+'_labels_chunk'+str(self.cur_chunk)+'.npy', mmap_mode='r+')
#         self.fnames = np.load(str(self.full_path) +'_'+self.subset+'_fnames_chunk'+str(self.cur_chunk)+'.npy', mmap_mode='r+')
        
#         self.data_tho = torch.from_numpy(self.data_tho)
#         self.data_mel = torch.from_numpy(self.data_mel)
        
#         self.n_files = np.shape(self.data_tho)[0]
#         self.n_tho = np.shape(self.data_tho)[2]
#         self.n_mels = np.shape(self.data_mel)[2]
#         self.n_mel_frames_per_file = np.shape(self.data_mel)[1]
        
#         self.len_cur_chun = len(self.fnames)
        
#     def _count_chunks(self):
#         count=0
#         for root, dirs, files in os.walk(self.dataset_path):
#             for file in files:
#                 if self.subset + '_third_octave_data_chunk' in file:
#                     count+=1
#         if count == 0:
#             raise Exception(f"No data found in folder {self.dataset_path}") 
#         return(count)
            
#     def __getitem__(self, idx):
        
#         idx_oracle = self.nb_data_processed + idx
        
#         input_tho = self.data_tho[idx]
#         input_mel = self.data_mel[idx]
#         input_fname = self.fnames[idx]
        
#         if self.oracle_predictions:
#             input_oracle_prediction = self.oracle_predictions[idx_oracle]
#         else:
#             input_oracle_prediction = np.array([])
    
#         return (input_tho , input_mel, input_fname, input_oracle_prediction)

#     def __len__(self):
#         return self.n_files


'''
OLD: commented on 22/11/2022
'''

'''
Functional dataloader without the oracle predictions
'''
# class MelDataset(torch.utils.data.Dataset):
#     def __init__(self, setting_data, n_tho_frames_per_file=8, subset='train', mels_type='YamNet', project_data_path=Path().absolute() ):

#         self.audio_dir = setting_data['data']['audio_dir']
#         self.dataset_name = setting_data['data']['dataset_name']

#         self.dataset_path = os.path.join(setting_data['data']['root_dir'], self.dataset_name)
#         self.dataset_path = project_data_path / self.dataset_path
        
#         self.sr = setting_data['data']['sr']
#         self.pad_examples = setting_data['data']['pad_examples']
#         self.flen = setting_data['data']['frame_length']
#         self.hlen = setting_data['data']['hop_length']
#         self.train_ratio = setting_data['data']['train_ratio']
#         self.valid_ratio = setting_data['data']['valid_ratio']
#         self.eval_ratio = setting_data['data']['eval_ratio']
#         self.dbtype = setting_data['data']['dbtype']
#         self.n_mels = setting_data['data']['n_mels']
#         self.labels_dict = setting_data['data']['labels_dict']
#         self.subset = subset
#         self.n_tho_frames_per_file = n_tho_frames_per_file

#         #mmap used to memory-map the file. It is stored in disk, but 
#         #small fragments of the file can be accessed without reading the
#         #entire file into memory. I removed mmap on data_tho and 
#         #data_mel because of the warning "UserWarning: The given NumPy array 
#         #is not writable"
#         self.full_path = self.dataset_path / self.dataset_name
#         self.cur_chunk = 0
#         self.n_chunks = self._count_chunks()
        
#         self.mels_type = mels_type
#         if mels_type == "PANN":
#             self.mels_root = 'mels_pann'
#         if mels_type == 'YamNet':
#             self.mels_root = 'mels_yamnet'
            
#         self.load_next_chunk(init=True)
        
#     def load_next_chunk(self, init=False):

#         if init:
#             self.cur_chunk=0
#         else:
#             self.cur_chunk+=1  
            
#         if self.cur_chunk >= self.n_chunks:
#             #print('no more chunks')
#             return
            
#         self.data_tho = np.load(str(self.full_path) +'_'+self.subset+'_third_octave_data_chunk'+str(self.cur_chunk)+'.npy', mmap_mode='r')
#         self.data_mel = np.load(str(self.full_path) +'_'+self.subset+'_'+self.mels_root+'_data_chunk'+str(self.cur_chunk)+'.npy', mmap_mode='r')
#         self.metadata = np.load(str(self.full_path) +'_'+self.subset+'_metadata_chunk'+str(self.cur_chunk)+'.npy', mmap_mode='r')
#         self.labels = np.load(str(self.full_path) +'_'+self.subset+'_labels_chunk'+str(self.cur_chunk)+'.npy', mmap_mode='r')
#         self.fnames = np.load(str(self.full_path) +'_'+self.subset+'_fnames_chunk'+str(self.cur_chunk)+'.npy', mmap_mode='r')
        
#         self.data_tho = torch.from_numpy(self.data_tho)
#         self.data_mel = torch.from_numpy(self.data_mel)
        
#         self.n_files = np.shape(self.data_tho)[0]
#         self.n_tho = np.shape(self.data_tho)[2]
#         self.n_mels = np.shape(self.data_mel)[2]
#         self.n_mel_frames_per_file = np.shape(self.data_mel)[1]
        
#     def _count_chunks(self):
#         count=0
#         for root, dirs, files in os.walk(self.dataset_path):
#             for file in files:
#                 if self.subset + '_third_octave_data_chunk' in file:
#                     count+=1
#         if count == 0:
#             raise Exception(f"No data found in folder {self.dataset_path}") 
#         return(count)
            
#     def __getitem__(self, idx):

#         input_tho = self.data_tho[idx]
#         input_mel = self.data_mel[idx]
#         input_fname = self.fnames[idx]
    
#         return (input_tho , input_mel, input_fname)

#     def __len__(self):
#         return self.n_files

# class OutputsDataset(torch.utils.data.Dataset):
#     def __init__(self, setting_data, model_name, mels_type='YamNet', project_data_path=Path().absolute()):
        
#         #mmap used to memory-map the file. It is stored in disk, but 
#         #small fragments of the file can be accessed without reading the
#         #entire file into memory. I removed mmap on data_tho and 
#         #data_mel because of the warning "UserWarning: The given NumPy array 
#         #is not writable"
#         self.dataset_name = setting_data['data']['dataset_name']
#         self.model_name = model_name
#         self.batch_type_name = mels_type
        
#         self.model_predictions_name = self.batch_type_name + '_' + self.model_name + '_' + self.dataset_name
#         self.model_predictions_path = project_data_path / 'outputs' / 'predictions' / model_name / self.model_predictions_name
        
#         self.model_predictions_filtered_name = self.batch_type_name + '_' + self.model_name + '_' + self.dataset_name
#         self.model_predictions_filtered_path = project_data_path / 'outputs' / 'predictions_filtered' / model_name / self.model_predictions_name
        
#         self.model_mels_name =  self.batch_type_name + '_' + self.model_name + '_' + self.dataset_name
#         self.model_mels_path = project_data_path / 'outputs' / 'mels' / model_name / self.model_mels_name
        
#         self.oracle_predictions_path =  project_data_path / 'outputs' / 'predictions' / 'oracle' / (self.batch_type_name + '_oracle_' + self.dataset_name)
#         self.oracle_predictions_name = self.batch_type_name + '_oracle_' + self.dataset_name
        
#         self.oracle_predictions_filtered_path =  project_data_path / 'outputs' / 'predictions_filtered' / 'oracle' / (self.batch_type_name + '_oracle_' + self.dataset_name)
#         self.oracle_predictions_filtered_name = self.batch_type_name + '_oracle_' + self.dataset_name
        
#         self.oracle_mels_path =  project_data_path / 'outputs' / 'mels' / 'oracle' / (self.batch_type_name + '_oracle_' + self.dataset_name)
#         self.oracle_mels_name = self.batch_type_name + '_oracle_' + self.dataset_name

#         # self.full_predictions_path = self.predictions_path / self.predictions_name
#         # self.full_mels_path = self.mels_path / self.mels_name

#         self.cur_chunk = 0
#         self.n_chunks = self._count_chunks()
            
#         self.load_next_chunk(init=True)
        
#     def load_next_chunk(self, init=False):

#         if init:
#             self.cur_chunk=0
#         else:
#             self.cur_chunk+=1  
            
#         if self.cur_chunk >= self.n_chunks:
#             #print('no more chunks')
#             return
        
#         self.oracle_predictions = np.load(self.oracle_predictions_path / (self.oracle_predictions_name + '_' + str(self.cur_chunk) +'.npy'), mmap_mode ='r')
#         self.oracle_predictions_filtered = np.load(self.oracle_predictions_filtered_path / (self.oracle_predictions_filtered_name + '_' + str(self.cur_chunk) +'.npy'), mmap_mode ='r')
#         self.oracle_mels = np.load(self.oracle_mels_path / (self.oracle_mels_name + '_' + str(self.cur_chunk) +'.npy'), mmap_mode ='r')

#         self.model_predictions = np.load(self.model_predictions_path / (self.model_predictions_name + '_' + str(self.cur_chunk) +'.npy'), mmap_mode ='r')
#         self.model_predictions_filtered = np.load(self.model_predictions_filtered_path / (self.model_predictions_filtered_name + '_' + str(self.cur_chunk) +'.npy'), mmap_mode ='r')
#         self.model_mels = np.load(self.model_mels_path / (self.model_mels_name + '_' + str(self.cur_chunk) +'.npy'), mmap_mode ='r')
        
#         self.oracle_predictions = torch.from_numpy(self.oracle_predictions)
#         self.oracle_predictions_filtered = torch.from_numpy(self.oracle_predictions_filtered)
#         self.oracle_mels = torch.from_numpy(self.oracle_mels)
        
#         self.model_predictions = torch.from_numpy(self.model_predictions)
#         self.model_predictions_filtered = torch.from_numpy(self.model_predictions_filtered)
#         self.model_mels = torch.from_numpy(self.model_mels)
                
#         self.n_files = np.shape(self.oracle_predictions)[0]
        
#     def _count_chunks(self):
#         count=0
#         for root, dirs, files in os.walk(self.model_predictions_path):
#             for file in files:
#                 count+=1
#         if count == 0:
#             raise Exception(f"No data found in folder {self.model_predictions_path}") 
#         return(count)
            
#     def __getitem__(self, idx):

#         oracle_prediction = self.oracle_predictions[idx]
#         oracle_prediction_filtered = self.oracle_predictions_filtered[idx]
#         oracle_mel = self.oracle_mels[idx]
        
#         model_prediction = self.model_predictions[idx]
#         model_prediction_filtered = self.model_predictions_filtered[idx]
#         model_mel = self.model_mels[idx]

#         return (oracle_prediction, oracle_prediction_filtered, oracle_mel, model_prediction, model_prediction_filtered, model_mel)

#     def __len__(self):
#         return self.n_files
'''
OLD
'''

# class MelDataset(torch.utils.data.Dataset):
#     def __init__(self, setting_data, n_tho_frames_per_file=8, subset='train'):

#         self.audio_dir = setting_data['data']['audio_dir']
#         self.dataset_name = setting_data['data']['dataset_name']

#         self.dataset_path = os.path.join(setting_data['data']['root_dir'], self.dataset_name)
#         self.dataset_path = Path().absolute() / self.dataset_path
        
#         self.sr = setting_data['data']['sr']
#         self.pad_examples = setting_data['data']['pad_examples']
#         self.flen = setting_data['data']['frame_length']
#         self.hlen = setting_data['data']['hop_length']
#         self.prop_train = setting_data['data']['prop_train']
#         self.prop_valid = setting_data['data']['prop_valid']
#         self.prop_eval = setting_data['data']['prop_eval']
#         self.dbtype = setting_data['data']['dbtype']
#         self.n_mels = setting_data['data']['n_mels']
#         self.labels_dict = setting_data['data']['labels_dict']
#         self.subset = subset
#         self.n_tho_frames_per_file = n_tho_frames_per_file

#         #mmap used to memory-map the file. It is stored in disk, but 
#         #small fragments of the file can be accessed without reading the
#         #entire file into memory. I removed mmap on data_tho and 
#         #data_mel because of the warning "UserWarning: The given NumPy array 
#         #is not writable"
#         self.full_path = self.dataset_path / self.dataset_name
#         self.cur_chunk = 0
#         self.n_chunks = self._count_chunks()
#         self.load_next_chunk(init=True)
        

        
#         #self.data_tho = np.load(self.dataset_path+'_'+subset+'_third_octave_data.npy', mmap_mode='r+')
#         #self.data_mel = np.load(self.dataset_path+'_'+subset+'_mel_data.npy', mmap_mode='r+')
#         #self.metadata = np.load(self.dataset_path+'_'+subset+'_metadata.npy', mmap_mode='r+')
#         #self.labels = np.load(self.dataset_path+'_'+subset+'_labels.npy', mmap_mode='r+')

#         # self.n_files = np.shape(self.data_tho)[0]
#         # self.n_tho_frames_per_file = np.shape(self.data_tho)[1]
#         # self.n_tho = np.shape(self.data_tho)[2]
#         # self.n_tho_frames = self.n_files * self.n_tho_frames_per_file
#         # self.n_tho_frames_per_file = n_tho_frames_per_file
#         # self.n_temporal_blocs = self.n_tho_frames // self.n_tho_frames_per_file
#         # self.n_blocs_per_frame = int(self.n_tho_frames_per_file / self.n_tho_frames_per_file)
        
#         # self.n_temporal_blocs_per_file = self.n_tho_frames_per_file // self.n_tho_frames_per_file
        
#         # self.n_mel_frames_per_file = np.shape(self.data_mel)[1]
#         # self.n_mels = np.shape(self.data_mel)[2]
#         # self.n_tho_frames_mels = self.n_files * self.n_mel_frames_per_file
#         # self.n_mel_frames_per_file = self.n_mel_frames_per_file // self.n_temporal_blocs_per_file
#         # self.n_temporal_blocs_mels = self.n_tho_frames_mels // self.n_mel_frames_per_file
#         # self.n_blocs_per_frame_mels = int(self.n_mel_frames_per_file / self.n_tho_frames_per_file)
    
#     def load_next_chunk(self, init=False):

#         if init:
#             self.cur_chunk=0
#         else:
#             self.cur_chunk+=1  
            
#         if self.cur_chunk >= self.n_chunks:
#             #print('no more chunks')
#             return
            
#         self.data_tho = np.load(str(self.full_path) +'_'+self.subset+'_third_octave_data_chunk'+str(self.cur_chunk)+'.npy', mmap_mode='r+')
#         self.data_mel = np.load(str(self.full_path) +'_'+self.subset+'_mel_data_chunk'+str(self.cur_chunk)+'.npy', mmap_mode='r+')
#         self.metadata = np.load(str(self.full_path) +'_'+self.subset+'_metadata_chunk'+str(self.cur_chunk)+'.npy', mmap_mode='r+')
#         self.labels = np.load(str(self.full_path) +'_'+self.subset+'_labels_chunk'+str(self.cur_chunk)+'.npy', mmap_mode='r+')
        
        
#         self.n_files = np.shape(self.data_tho)[0]
#         self.n_tho_frames_per_file = np.shape(self.data_tho)[1]
#         self.n_tho = np.shape(self.data_tho)[2]
        
        
#         self.n_tho_frames = self.n_files * self.n_tho_frames_per_file
#         self.n_temporal_blocs = self.n_tho_frames // self.n_tho_frames_per_file
#         self.n_blocs_per_frame = int(self.n_tho_frames_per_file / self.n_tho_frames_per_file)
        
#         self.n_temporal_blocs_per_file = self.n_tho_frames_per_file // self.n_tho_frames_per_file
        
#         self.n_mel_frames_per_file = np.shape(self.data_mel)[1]
#         self.n_mels = np.shape(self.data_mel)[2]
#         self.n_tho_frames_mels = self.n_files * self.n_mel_frames_per_file
#         self.n_mel_frames_per_file = self.n_mel_frames_per_file // self.n_temporal_blocs_per_file
#         self.n_temporal_blocs_mels = self.n_tho_frames_mels // self.n_mel_frames_per_file
#         self.n_blocs_per_frame_mels = int(self.n_mel_frames_per_file / self.n_tho_frames_per_file)
        
#     def _count_chunks(self):
#         count=0
#         for root, dirs, files in os.walk(self.dataset_path):
#             for file in files:
#                 if self.subset + '_third_octave_data_chunk' in file:
#                     count+=1
#         if count == 0:
#             raise Exception(f"No data found in folder {self.dataset_path}") 
#         return(count)
            
#     def __getitem__(self, idx):
        
#         q = idx // self.n_blocs_per_frame
#         r = idx % self.n_blocs_per_frame

#         input_tho = torch.from_numpy(self.data_tho[q][r*self.n_tho_frames_per_file: (r+1)*self.n_tho_frames_per_file])
#         input_mel = torch.from_numpy(self.data_mel[q][r*self.n_mel_frames_per_file: (r+1)*self.n_mel_frames_per_file+1])

#         return (input_tho , input_mel)

#     def __len__(self):
#         return self.n_temporal_blocs
    


    
# class MelDataset(torch.utils.data.Dataset):
#     def __init__(self, setting_data, n_tho_frames_per_file=8, subset='train'):

#         self.audio_dir = setting_data['data']['audio_dir']
#         self.dataset_name = setting_data['data']['dataset_name']

#         self.dataset_path = os.path.join(setting_data['data']['root_dir'], self.dataset_name)
#         self.dataset_path = Path().absolute() / self.dataset_path
        
#         self.sr = setting_data['data']['sr']
#         self.pad_examples = setting_data['data']['pad_examples']
#         self.flen = setting_data['data']['frame_length']
#         self.hlen = setting_data['data']['hop_length']
#         self.prop_train = setting_data['data']['prop_train']
#         self.prop_valid = setting_data['data']['prop_valid']
#         self.prop_eval = setting_data['data']['prop_eval']
#         self.dbtype = setting_data['data']['dbtype']
#         self.n_mels = setting_data['data']['n_mels']
#         self.labels_dict = setting_data['data']['labels_dict']
        
#         #mmap used to memory-map the file. It is stored in disk, but 
#         #small fragments of the file can be accessed without reading the
#         #entire file into memory. I removed mmap on data_tho and 
#         #data_mel because of the warning "UserWarning: The given NumPy array 
#         #is not writable"
#         self.full_path = dataset_path / dataset_name
#         self.data_tho = np.load(str(self.dataset_path) +'_'+subset+'_third_octave_data.npy', mmap_mode='r+')
#         self.data_mel = np.load(str(self.dataset_path) +'_'+subset+'_mel_data.npy', mmap_mode='r+')
#         self.metadata = np.load(str(self.dataset_path) +'_'+subset+'_metadata.npy', mmap_mode='r+')
#         self.labels = np.load(str(self.dataset_path) +'_'+subset+'_labels.npy', mmap_mode='r+')
        
#         #self.data_tho = np.load(self.dataset_path+'_'+subset+'_third_octave_data.npy', mmap_mode='r+')
#         #self.data_mel = np.load(self.dataset_path+'_'+subset+'_mel_data.npy', mmap_mode='r+')
#         #self.metadata = np.load(self.dataset_path+'_'+subset+'_metadata.npy', mmap_mode='r+')
#         #self.labels = np.load(self.dataset_path+'_'+subset+'_labels.npy', mmap_mode='r+')
        
        

#         self.n_files = np.shape(self.data_tho)[0]
#         self.n_tho_frames_per_file = np.shape(self.data_tho)[1]
#         self.n_tho = np.shape(self.data_tho)[2]
#         self.n_tho_frames = self.n_files * self.n_tho_frames_per_file
#         self.n_tho_frames_per_file = n_tho_frames_per_file
#         self.n_temporal_blocs = self.n_tho_frames // self.n_tho_frames_per_file
#         self.n_blocs_per_frame = int(self.n_tho_frames_per_file / self.n_tho_frames_per_file)
        
#         self.n_temporal_blocs_per_file = self.n_tho_frames_per_file // self.n_tho_frames_per_file
        
#         self.n_mel_frames_per_file = np.shape(self.data_mel)[1]
#         self.n_mels = np.shape(self.data_mel)[2]
#         self.n_tho_frames_mels = self.n_files * self.n_mel_frames_per_file
#         self.n_mel_frames_per_file = self.n_mel_frames_per_file // self.n_temporal_blocs_per_file
#         self.n_temporal_blocs_mels = self.n_tho_frames_mels // self.n_mel_frames_per_file
#         self.n_blocs_per_frame_mels = int(self.n_mel_frames_per_file / self.n_tho_frames_per_file) 

#     def __getitem__(self, idx):
        
#         q = idx // self.n_blocs_per_frame
#         r = idx % self.n_blocs_per_frame

#         input_tho = torch.from_numpy(self.data_tho[q][r*self.n_tho_frames_per_file: (r+1)*self.n_tho_frames_per_file])
#         input_mel = torch.from_numpy(self.data_mel[q][r*self.n_mel_frames_per_file: (r+1)*self.n_mel_frames_per_file+1])

#         return (input_tho , input_mel)

#     def __len__(self):
#         return self.n_temporal_blocs

'''
OLD: commented on 22/11/2022
'''
# class PinvDataset(torch.utils.data.Dataset):
#     def __init__(self, setting_data, n_tho_frames_per_file=8, subset='train'):

#         self.audio_dir = setting_data['data']['audio_dir']
#         self.dataset_name = setting_data['data']['dataset_name']

#         self.dataset_path = os.path.join(setting_data['data']['root_dir'], self.dataset_name)
#         self.dataset_path = Path().absolute() / self.dataset_path
        
#         self.sr = setting_data['data']['sr']
#         self.pad_examples = setting_data['data']['pad_examples']
#         self.flen = setting_data['data']['frame_length']
#         self.hlen = setting_data['data']['hop_length']
#         self.prop_train = setting_data['data']['prop_train']
#         self.prop_valid = setting_data['data']['prop_valid']
#         self.prop_eval = setting_data['data']['prop_eval']
#         self.dbtype = setting_data['data']['dbtype']
#         self.n_mels = setting_data['data']['n_mels']
#         self.labels_dict = setting_data['data']['labels_dict']
        
#         #mmap used to memory-map the file. It is stored in disk, but 
#         #small fragments of the file can be accessed without reading the
#         #entire file into memory. I removed mmap on data_tho and 
#         #data_mel because of the warning "UserWarning: The given NumPy array 
#         #is not writable"
        
#         self.data_tho = np.load(str(self.dataset_path) +'_'+subset+'_third_octave_data.npy', mmap_mode='r+')
#         self.data_mel = np.load(str(self.dataset_path) +'_'+subset+'_mel_data.npy', mmap_mode='r+')
#         self.metadata = np.load(str(self.dataset_path) +'_'+subset+'_metadata.npy', mmap_mode='r+')
#         self.labels = np.load(str(self.dataset_path) +'_'+subset+'_labels.npy', mmap_mode='r+')
        
#         #self.data_tho = np.load(self.dataset_path+'_'+subset+'_third_octave_data.npy', mmap_mode='r+')
#         #self.data_mel = np.load(self.dataset_path+'_'+subset+'_mel_data.npy', mmap_mode='r+')
#         #self.metadata = np.load(self.dataset_path+'_'+subset+'_metadata.npy', mmap_mode='r+')
#         #self.labels = np.load(self.dataset_path+'_'+subset+'_labels.npy', mmap_mode='r+')
        
        

#         self.n_files = np.shape(self.data_tho)[0]
#         self.n_tho_frames_per_file = np.shape(self.data_tho)[1]
#         self.n_tho = np.shape(self.data_tho)[2]
#         self.n_tho_frames = self.n_files * self.n_tho_frames_per_file
#         self.n_tho_frames_per_file = n_tho_frames_per_file
#         self.n_temporal_blocs = self.n_tho_frames // self.n_tho_frames_per_file
#         self.n_blocs_per_frame = int(self.n_tho_frames_per_file / self.n_tho_frames_per_file)
        
#         self.n_temporal_blocs_per_file = self.n_tho_frames_per_file // self.n_tho_frames_per_file
        
#         self.n_mel_frames_per_file = np.shape(self.data_mel)[1]
#         self.n_mels = np.shape(self.data_mel)[2]
#         self.n_tho_frames_mels = self.n_files * self.n_mel_frames_per_file
#         self.n_mel_frames_per_file = self.n_mel_frames_per_file // self.n_temporal_blocs_per_file
#         self.n_temporal_blocs_mels = self.n_tho_frames_mels // self.n_mel_frames_per_file
#         self.n_blocs_per_frame_mels = int(self.n_mel_frames_per_file / self.n_tho_frames_per_file)
        

#     def __getitem__(self, idx):
        
#         input_tho = self.data_tho[idx]
#         input_mel = self.data_mel[idx]

#         return (input_tho , input_mel)

#     def __len__(self):
#         return self.n_files
    
# class TestingDataset(torch.utils.data.Dataset):
#     def __init__(self, setting_data, n_tho_frames_per_file=8, subset='train'):

#         self.audio_dir = setting_data['data']['audio_dir']
#         self.dataset_name = setting_data['data']['dataset_name']

#         self.dataset_path = os.path.join(setting_data['data']['root_dir'], self.dataset_name)
#         self.dataset_path = Path().absolute() / self.dataset_path
        
#         self.sr = setting_data['data']['sr']
#         self.pad_examples = setting_data['data']['pad_examples']
#         self.flen = setting_data['data']['frame_length']
#         self.hlen = setting_data['data']['hop_length']
#         self.prop_train = setting_data['data']['prop_train']
#         self.prop_valid = setting_data['data']['prop_valid']
#         self.prop_eval = setting_data['data']['prop_eval']
#         self.dbtype = setting_data['data']['dbtype']
#         self.n_mels = setting_data['data']['n_mels']
#         self.labels_dict = setting_data['data']['labels_dict']
        
#         #mmap used to memory-map the file. It is stored in disk, but 
#         #small fragments of the file can be accessed without reading the
#         #entire file into memory. I removed mmap on data_tho and 
#         #data_mel because of the warning "UserWarning: The given NumPy array 
#         #is not writable"
        
#         self.data_tho = np.load(str(self.dataset_path) +'_'+subset+'_third_octave_data.npy', mmap_mode='r+')
#         self.data_mel = np.load(str(self.dataset_path) +'_'+subset+'_mel_data.npy', mmap_mode='r+')
#         self.metadata = np.load(str(self.dataset_path) +'_'+subset+'_metadata.npy', mmap_mode='r+')
#         self.labels = np.load(str(self.dataset_path) +'_'+subset+'_labels.npy', mmap_mode='r+')
        
#         #self.data_tho = np.load(self.dataset_path+'_'+subset+'_third_octave_data.npy', mmap_mode='r+')
#         #self.data_mel = np.load(self.dataset_path+'_'+subset+'_mel_data.npy', mmap_mode='r+')
#         #self.metadata = np.load(self.dataset_path+'_'+subset+'_metadata.npy', mmap_mode='r+')
#         #self.labels = np.load(self.dataset_path+'_'+subset+'_labels.npy', mmap_mode='r+')
        
        

#         self.n_files = np.shape(self.data_tho)[0]
#         self.n_tho_frames_per_file = np.shape(self.data_tho)[1]
#         self.n_tho = np.shape(self.data_tho)[2]
#         self.n_tho_frames = self.n_files * self.n_tho_frames_per_file
#         self.n_tho_frames_per_file = n_tho_frames_per_file
#         self.n_temporal_blocs = self.n_tho_frames // self.n_tho_frames_per_file
#         self.n_blocs_per_frame = int(self.n_tho_frames_per_file / self.n_tho_frames_per_file)
        
#         self.n_temporal_blocs_per_file = self.n_tho_frames_per_file // self.n_tho_frames_per_file
        
#         self.n_mel_frames_per_file = np.shape(self.data_mel)[1]
#         self.n_mels = np.shape(self.data_mel)[2]
#         self.n_tho_frames_mels = self.n_files * self.n_mel_frames_per_file
#         self.n_mel_frames_per_file = self.n_mel_frames_per_file // self.n_temporal_blocs_per_file
#         self.n_temporal_blocs_mels = self.n_tho_frames_mels // self.n_mel_frames_per_file
#         self.n_blocs_per_frame_mels = int(self.n_mel_frames_per_file / self.n_tho_frames_per_file)
        

#     def __getitem__(self, idx):
        
#         q = idx // self.n_blocs_per_frame
#         r = idx % self.n_blocs_per_frame

#         input_tho = torch.from_numpy(self.data_tho[q][r*self.n_tho_frames_per_file: (r+1)*self.n_tho_frames_per_file])
#         input_mel = torch.from_numpy(self.data_mel[q][r*self.n_mel_frames_per_file: (r+1)*self.n_mel_frames_per_file+1])
        
#         #input_metadata = self.metadata[q]
#         #print('METADATA')
#         #print(input_metadata)

#         return (input_tho , input_mel)

#     def __len__(self):
#         return self.n_temporal_blocs