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