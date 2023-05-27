#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 17:45:28 2022

@author: user
"""

import torch
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import utils.pinv_transcoder as pt
from pann.pann_mel_inference import PannMelInference
from yamnet.yamnet_mel_inference import YamnetMelInference

class TSEvaluater:
    def __init__(self, setting_data, model, models_path, model_name, 
                 outputs_path, eval_dataset, dtype=torch.FloatTensor):
        """
        Initializes the TSEvaluater class. This saves inferences on logits 
        for teacher-student models that are not transcoders (effnet_b0,
        effnet_b7, self)

        Args:
        - setting_data: The setting data for the evaluation.
        - model: The pre-trained model.
        - models_path: The path to the model files.
        - model_name: The name of the model file.
        - outputs_path: The path to save the output files.
        - eval_dataset: The evaluation dataset.
        - dtype: The data type for the model (default: torch.FloatTensor).
        """
        self.outputs_path = outputs_path
        
        self.setting_data = setting_data
        self.dtype = dtype

        self.eval_dataset = eval_dataset
        
        self.model = model
        
        self.lr = 1e-3
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)
        
        self.models_path = models_path
        self.model_name = model_name
        
        self.classifier = self.eval_dataset.classifier
        
    def evaluate(self, batch_size=64, device=torch.device("cpu")):
        self.model.eval
        
        classifier = self.eval_dataset.classifier
        if classifier == 'PANN':
            self.classif_inference = PannMelInference(device=device)
        if classifier == 'YamNet':
            self.classif_inference = YamnetMelInference(device=device)
                
        self.eval_dataloader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
        tqdm_it=tqdm(self.eval_dataloader, desc='EVALUATION: Chunk {}/{}'.format(0,0))
        
        #save the output of the model in a .dat file. This avoids havind memory issues
        output_logits = np.memmap(self.outputs_path['logits']+'.dat', dtype=np.float64,
                      mode='w+', shape=(self.eval_dataset.n_tho_frames, self.classif_inference.n_labels))
        
        for (idx,x,y,file) in tqdm_it:
            x = x.type(self.dtype)
            y = y.type(self.dtype)
            
            x = x.to(device)
            y = y.to(device)
            
            y = torch.unsqueeze(y, 1)

            inf_classifier_pred = self.model(x)
            
            #labels = self.classif_inference.labels_tvb_enc
            
            output_logits[idx, :] = inf_classifier_pred.detach().cpu().numpy()
            
            output_logits.flush()
        
        #to open a .dat file as a npy array, the shape of the npy is needed
        np.save(self.outputs_path['logits']+'_shape.npy', output_logits.shape)
        
        return()
    
    def load_model(self, device):
        self.model = self.model.to(device)
        state_dict = torch.load(self.models_path + self.model_name, map_location=device)
        self.model.load_state_dict(state_dict)

class DLTranscoderEvaluater:
    def __init__(self, setting_data, model, models_path, model_name, 
                 outputs_path, eval_dataset, dtype=torch.FloatTensor):
        """
        Initializes the DLTranscoderEvaluater class. This class is used to 
        evaluate transcoders. It thus saves two dat files: one for the
        inference of the Mels, and one for the inference of the logits.
        This evaluation is used for mlp, mlp_pinv and cnn_pinv models.

        Args:
        - setting_data: The setting data for the evaluation.
        - model: The pre-trained model.
        - models_path: The path to the model files.
        - model_name: The name of the model file.
        - outputs_path: The path to save the output files.
        - eval_dataset: The evaluation dataset.
        - dtype: The data type for the model (default: torch.FloatTensor).
        """
        self.outputs_path = outputs_path
        
        self.setting_data = setting_data
        self.dtype = dtype

        self.eval_dataset = eval_dataset
        
        self.model = model
        
        self.lr = 1e-3
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)
        
        self.models_path = models_path
        self.model_name = model_name
                
        self.classifier = self.eval_dataset.classifier
    
    def evaluate(self, batch_size=64, device=torch.device("cpu")):

        self.model.eval
        
        classifier = self.eval_dataset.classifier
        if classifier == 'PANN':
            self.classif_inference = PannMelInference(device=device)
        if classifier == 'YamNet':
            self.classif_inference = YamnetMelInference(device=device)
                
        self.eval_dataloader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
        tqdm_it=tqdm(self.eval_dataloader, desc='EVALUATION: Chunk {}/{}'.format(0,0))
        
        output_mels = np.memmap(self.outputs_path['mels']+'.dat', dtype=np.float64,
                      mode='w+', shape=(self.eval_dataset.n_tho_frames,
                      self.eval_dataset.n_mel_frames_per_file, self.eval_dataset.n_mels))
        
        output_logits = np.memmap(self.outputs_path['logits']+'.dat', dtype=np.float64,
                      mode='w+', shape=(self.eval_dataset.n_tho_frames, self.classif_inference.n_labels))
                      
        for (idx,x,y,file) in tqdm_it:
            x = x.type(self.dtype)
            y = y.type(self.dtype)
            
            x = x.to(device)
            y = y.to(device)
            
            y = torch.unsqueeze(y, 1)

            y_pred = self.model(x)
            
            output_mels[idx, :, :] = y_pred.detach().cpu().numpy()
            
            y_pred = torch.unsqueeze(y_pred, 1)

            inf_classifier_pred = self.classif_inference.simple_inference(y_pred, filter_classes=True, softmax=False)
            
            #labels = self.classif_inference.labels_tvb_enc
            
            output_logits[idx, :] = inf_classifier_pred.detach().cpu().numpy()
            
            output_mels.flush()
            output_logits.flush()
            
        np.save(self.outputs_path['mels']+'_shape.npy', output_mels.shape)
        np.save(self.outputs_path['logits']+'_shape.npy', output_logits.shape)
        
        return()
    
    def load_model(self, device):
        self.model = self.model.to(device)
        state_dict = torch.load(self.models_path + self.model_name, map_location=device)
        self.model.load_state_dict(state_dict)
        
#for Pinv only. Not a real trainer, just an evaluater.
class PinvEvaluater:
    def __init__(self, eval_dataset, tho_tr, mels_tr, 
                 outputs_path, dtype=torch.FloatTensor):
        """
        Initializes the PinvEvaluater class. This evaluates the given dataset with PINV model.

        Args:
        - eval_dataset: The evaluation dataset.
        - tho_tr: The third octave transform used to transform audio to third-octaves.
        - mels_tr: The Mel transforma used to transform audio to Mels.
        - outputs_path: The path to save the output files.
        - dtype: The data type for the model (default: torch.FloatTensor).
        """
        self.outputs_path = outputs_path
        
        self.dtype = dtype

        self.eval_dataset = eval_dataset
        
        self.tho_tr = tho_tr
        self.mels_tr = mels_tr
        
    def evaluate(self, batch_size=64, device=torch.device("cpu"), oracle_predictions=None):
        
        self.device = device
        classifier = self.eval_dataset.classifier
        if classifier == 'PANN':
            self.classif_inference = PannMelInference(device=device)
        if classifier == 'YamNet':
            self.classif_inference = YamnetMelInference(device=device)

        self.eval_dataloader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
        tqdm_it=tqdm(self.eval_dataloader, desc='EVALUATION: Chunk {}/{}'.format(0,0))
        
        output_mels = np.memmap(self.outputs_path['mels']+'.dat', dtype=np.float64,
                      mode='w+', shape=(self.eval_dataset.n_tho_frames,
                      self.eval_dataset.n_mel_frames_per_file, self.eval_dataset.n_mels))
        
        output_logits = np.memmap(self.outputs_path['logits']+'.dat', dtype=np.float64,
                      mode='w+', shape=(self.eval_dataset.n_tho_frames, self.classif_inference.n_labels))
        
        for (idx, x,y,file) in tqdm_it:

            x = x.type(self.dtype)
            y = y.type(self.dtype)

            x = x.to(self.device)
            y = y.to(self.device)
            
            y = torch.unsqueeze(y, 1)
            
            y_pred = pt.pinv(x, self.tho_tr, self.mels_tr, reshape=y.shape[2], device=self.device)
            
            # FROM PREVIOUS VERSION: allowed to train the models on non-dB scaled spectrograms. This gave extremely poor results.
            # # TRUE EVALUATION
            # if not self.input_is_db:
            #     # y_pred = self.mels_tr.db_to_power(y_pred).requires_grad_()
            #     y_pred = self.mels_tr.power_to_db(y_pred)
            
            output_mels[idx, :, :] = y_pred.detach().cpu().numpy()
            
            y_pred = torch.squeeze(y_pred, 0)
            y_pred = torch.unsqueeze(y_pred, 1)

            inf_classifier_pred = self.classif_inference.simple_inference(y_pred, filter_classes=True, softmax=False)
            
            output_logits[idx, :] = inf_classifier_pred.detach().cpu().numpy()
            
            output_mels.flush()
            output_logits.flush()

        
        np.save(self.outputs_path['mels']+'_shape.npy', output_mels.shape)
        np.save(self.outputs_path['logits']+'_shape.npy', output_logits.shape)
        
        return()

class OracleEvaluater:
    def __init__(self, eval_dataset, 
                 outputs_path, dtype=torch.FloatTensor, tvb=False, label=None):
        """
        Initializes the OracleEvaluater class. The OracleEvaluater is used to
        make inference of Mels and logits with the groundtruth PANN or YamNet model.
        Those inferences are used to train the deep learning models.

        Args:
        - eval_dataset: The evaluation dataset.
        - outputs_path: The path to save the output files.
        - dtype: The data type for the model (default: torch.FloatTensor).
        - tvb: Whether to also save tvb (traffic, voices, birds) predictions in a specific array. 
                This would be used to train the teacher-student models only on tvb classes instead of
                 on the 527 (or 521) classes (default: False).
        - label: The label for the output files. If the results of the OracleEvaluater are used for training, validating and 
                evaluating the models, use 'train', 'valid' or 'eval' instead of None.
        """

        self.outputs_path = outputs_path
        
        self.dtype = dtype
        self.eval_dataset = eval_dataset
        
        self.tvb = tvb
        self.label = label

    
    def evaluate(self, batch_size=64, device=torch.device("cpu")):
        
        self.device = device
        classifier = self.eval_dataset.classifier
        if classifier == 'PANN':
            self.classif_inference = PannMelInference(device=device)
        if classifier == 'YamNet':
            self.classif_inference = YamnetMelInference(device=device)

        self.eval_dataloader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
        tqdm_it=tqdm(self.eval_dataloader, desc='EVALUATION: ')
        
        if self.label is None:
            add_str = ''
        else:
            add_str = '_'+self.label

        output_mels = np.memmap(self.outputs_path['mels']+add_str+'.dat', dtype=np.float64,
                    mode='w+', shape=(self.eval_dataset.n_tho_frames,
                    self.eval_dataset.n_mel_frames_per_file, self.eval_dataset.n_mels))
        
        output_logits = np.memmap(self.outputs_path['logits']+add_str+'.dat', dtype=np.float64,
                    mode='w+', shape=(self.eval_dataset.n_tho_frames, self.classif_inference.n_labels))

        if self.tvb:
            output_logits_tvb = np.memmap(self.outputs_path['logits_tvb']+add_str+'.dat', dtype=np.float64,
                    mode='w+', shape=(self.eval_dataset.n_tho_frames, self.classif_inference.n_labels_tvb))

        for (idx, x, y, _) in tqdm_it:

            x = x.type(self.dtype)
            y = y.type(self.dtype)

            x = x.to(self.device)
            y = y.to(self.device)

            output_mels[idx, :, :] = y.detach().cpu().numpy()
            
            y = torch.unsqueeze(y, 1)

            inf_classifier_pred, inf_classifier_filtered_pred = self.classif_inference.inference(y, filter_classes=True, softmax=False)
            
            output_logits[idx, :] = inf_classifier_pred.detach().cpu().numpy()
            if self.tvb:
                output_logits_tvb[idx, :] = inf_classifier_filtered_pred.detach().cpu().numpy()
            
            output_mels.flush()
            output_logits.flush()
            if self.tvb:
                output_logits_tvb.flush()
        
        if self.label is None:
            np.save(self.outputs_path['mels']+'_shape.npy', output_mels.shape)
            np.save(self.outputs_path['logits']+'_shape.npy', output_logits.shape)
            if self.tvb:
                np.save(self.outputs_path['logits_tvb']+'_shape.npy', output_logits_tvb.shape)
        else:
            np.save(self.outputs_path['mels']+'_'+self.label+'_shape.npy', output_mels.shape)
            np.save(self.outputs_path['logits']+'_'+self.label+'_shape.npy', output_logits.shape)
            if self.tvb:
                np.save(self.outputs_path['logits_tvb']+'_'+self.label+'_shape.npy', output_logits_tvb.shape)
        return()
    

