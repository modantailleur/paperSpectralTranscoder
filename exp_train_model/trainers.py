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
import time
import torch.nn as nn
import copy
import yaml
import utils.util as ut
from pann.pann_mel_inference import PannMelInference
from yamnet.yamnet_mel_inference import YamnetMelInference

class ParameterCountCallback:
    def __init__(self, model):
        self.model = model
        self.updated_params = 0

    def on_batch_end(self, batch, logs=None):
        self.updated_params = 0
        for param in self.model.parameters():
            if param.grad is not None:
                self.updated_params += torch.sum(param.grad != 0).item()
        print(f"Iteration {batch}: Updated {self.updated_params} parameters")

class HybridTrainer:
    def __init__(self, setting_data, model, models_path, transcoder, model_name, train_dataset=None, 
                 valid_dataset=None, eval_dataset=None, learning_rate=1e-3, dtype=torch.FloatTensor, 
                 ltype=torch.LongTensor, classifier='PANN', prop_logit=100):
        """
        Initializes a HybridTrainer. The HybridTrainer trains a model both on Mels and Logits values, unless prop_logit
        is set to 100 (default). If prop_logit is set to 100, the model is trained only on logit values.

        Args:
        - setting_data: A dictionary containing various setting_data. This dictionnary isn't useful for the trainer, but is stored in its own settings.
        - model: The transcoder to be trained.
        - models_path: The path where the trained models will be saved.
        - transcoder (str): The type of transcoder (mlp, mlp_pinv, cnn_pinv)
        - model_name: The name of the model.
        - train_dataset: The dataset used for training the model. (default: None)
        - valid_dataset: The dataset used for validation during training. (default: None)
        - eval_dataset: The dataset used for evaluation. (default: None)
        - learning_rate: The learning rate for the optimizer. (default: 1e-3)
        - dtype: The data type used for the model. (default: torch.FloatTensor)
        - ltype: The data type used for labels. (default: torch.LongTensor)
        - classifier: The type of classifier used. (default: 'PANN')
        - prop_logit: The proportion of logit loss compared to mel loss. (default: 100)
        """

        #factors in front of loss functions, if a hybrid training method is chosen
        if prop_logit != 100:
            if classifier == 'PANN':
                self.k_mel = 1*(1-prop_logit/100)
                self.k_logit = 1500 * prop_logit / 100
            if classifier == 'YamNet':
                self.k_mel = 1*(1-prop_logit/100)
                self.k_logit = 10 * prop_logit / 100

        self.train_duration = 0

        self.prop_logit = prop_logit
        self.setting_data = setting_data
        self.dtype = dtype
        self.ltype = ltype
        self.classifier = classifier
        
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.eval_dataset = eval_dataset
        
        for dataset in [train_dataset, valid_dataset, eval_dataset]:
            if dataset != None:
                self.flen = dataset.flen
                self.hlen = dataset.hlen
                self.sr = dataset.sr
                break
        
        self.model = model
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)
        
        self.models_path = models_path
        self.model_name = model_name
        self.transcoder = transcoder

        #for validation (not necessary if first validation is set before train)
        self.best_loss = float('inf')
        self.best_state_dict = copy.deepcopy(self.model.state_dict())
        self.best_epoch = -1

        print('TRAINED MODEL')
        ut.count_parameters(self.model)
        
    def train(self, batch_size=64, epochs=10, device=torch.device("cpu")):
        
        mels_type = self.eval_dataset.mels_type
        if mels_type == 'PANN':
            self.classif_inference = PannMelInference(device=device)
        if mels_type == 'YamNet':
            self.classif_inference = YamnetMelInference(device=device)
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device

        losses_train = []
        losses_valid = []
        losses_eval = []

        losses_mel_train = []
        losses_mel_valid = []
        losses_mel_eval = []

        losses_logit_train = []
        losses_logit_valid = []
        losses_logit_eval = []

        self.model.train()
        self.loss_function_mel = nn.MSELoss()
        self.loss_function_logit = nn.BCELoss()

        self.model = self.model.to(self.device)

        #fist validation, with random parameters for the transcoder
        loss_valid, loss_mel_valid, loss_logit_valid = self.validate(self.valid_dataset, 0, batch_size=batch_size, device=self.device, forced=True)
        losses_valid.append(loss_valid)
        
        loss_eval, loss_mel_eval, loss_logit_eval = self.validate(self.eval_dataset, 0, batch_size=batch_size, device=self.device, label='EVALUATION')
        losses_eval.append(loss_eval)

        #print the number of parameters of the trained model
        #ParameterCountCallback(self.model)

        #validation on evaluation
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False, drop_last=False)
        cur_loss = 0
        for cur_epoch in range(self.epochs):

            #different tqdm display if hybrid training or logit training
            if self.prop_logit !=100:
                tqdm_it=tqdm(self.train_dataloader, 
                            desc='TRAINING: Epoch {}, Chunk {}/{}, loss: {:.4f}, loss_mel: {:.4f}, loss_logit: {:.4f}'
                            .format(cur_epoch+1, 0, 0, cur_loss, 0, 0))
            else:
                tqdm_it=tqdm(self.train_dataloader, 
                            desc='TRAINING: Epoch {}, Chunk {}/{}, loss: {:.4f}'
                            .format(cur_epoch + 1, 0, 0, cur_loss), mininterval=0)
                            
            for (idx, x, y, input_fname, oracle_prediction, oracle_prediction_filtered) in tqdm_it:

                start_time = time.time()

                x = x.type(self.dtype)
                y = y.type(self.dtype)
                oracle_prediction_filtered = oracle_prediction_filtered.type(self.dtype)
                oracle_prediction = oracle_prediction.type(self.dtype)

                x = x.to(self.device)
                y = y.to(self.device)

                oracle_prediction_filtered = oracle_prediction_filtered.to(self.device)
                oracle_prediction = oracle_prediction.to(self.device)

                self.optimizer.zero_grad()
                
                y_pred = self.model(x)
                y_pred = torch.unsqueeze(y_pred, 1)
                y = torch.unsqueeze(y, 1)
                
                inf_classifier_pred = self.classif_inference.simple_inference(y_pred, filter_classes=True, softmax=False, no_grad=False)

                # duration = time.time() - start_time
                # print(f'duration backward: {duration}')

                # duration = time.time() - start_time
                # print(f'duration pann inference: {duration}')

                if self.prop_logit != 100:
                    cur_mel_loss = self.k_mel*self.loss_function_mel(y_pred,y)
                    cur_logit_loss = self.k_logit*self.loss_function_logit(inf_classifier_pred, oracle_prediction)
                    cur_loss = cur_mel_loss + cur_logit_loss
                else:
                    cur_loss = self.loss_function_logit(inf_classifier_pred, oracle_prediction)

                cur_loss.backward()
                batch_duration = time.time() - start_time
                self.train_duration += batch_duration
                self.optimizer.step()
                
                cur_loss = float(cur_loss.data)

                if self.prop_logit != 100:
                    cur_mel_loss = float(cur_mel_loss.data)
                    cur_logit_loss = float(cur_logit_loss.data)
                
                losses_train.append(cur_loss)

                if self.prop_logit != 100:
                    losses_mel_train.append(cur_mel_loss)
                    losses_logit_train.append(cur_logit_loss)

                if self.prop_logit != 100:
                    tqdm_it.set_description('TRAINING: Epoch {}, Chunk {}/{}, loss: {:.4f}, loss_mel: {:.4f}, loss_logit: {:.4f}'
                                            .format(cur_epoch+1,0,0,cur_loss, cur_mel_loss, cur_logit_loss))
                else:
                    tqdm_it.set_description('TRAINING: Epoch {}, Chunk {}/{}, loss: {:.4f}'
                        .format(cur_epoch + 1, 0, 0, cur_loss))
          
                
            #Validation
            loss_valid, loss_mel_valid, loss_logit_valid = self.validate(self.valid_dataset, cur_epoch, batch_size=batch_size, device=self.device)
            losses_valid.append(loss_valid)
            losses_mel_valid.append(loss_mel_valid)
            losses_logit_valid.append(loss_logit_valid)

            loss_eval, loss_mel_eval, loss_logit_eval = self.validate(self.eval_dataset, cur_epoch, batch_size=batch_size, device=self.device, label='EVALUATION')
            losses_eval.append(loss_eval)
            losses_mel_eval.append(loss_mel_eval)
            losses_logit_eval.append(loss_logit_eval)

        losses = {
                'losses_train': np.array(losses_train),
                'losses_valid': np.array(losses_valid),
                'losses_eval': np.array(losses_eval),
                'losses_mel_train': np.array(losses_mel_train),
                'losses_mel_valid': np.array(losses_mel_valid),
                'losses_mel_eval': np.array(losses_mel_eval),
                'losses_logit_train': np.array(losses_logit_train),
                'losses_logit_valid': np.array(losses_logit_valid),
                'losses_logit_eval': np.array(losses_logit_eval)
            }
        
        return(losses)
    
    def validate(self, dataset, cur_epoch, batch_size=64, device=torch.device("cpu"), label='VALIDATION', forced=False):
        self.model.eval
        losses_valid = []
        losses_mel_valid = []
        losses_logit_valid = []

        #no need to shuffle during validation
        valid_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
        tqdm_it=tqdm(valid_dataloader, desc=label+': Chunk {}/{}'.format(0,0))
        for (idx, x , y, input_fname, oracle_prediction, oracle_prediction_filtered) in tqdm_it:
            x = x.type(self.dtype)
            y = y.type(self.dtype)
            oracle_prediction_filtered = oracle_prediction_filtered.type(self.dtype)
            oracle_prediction = oracle_prediction.type(self.dtype)

            x = x.to(device)
            y = y.to(device)
            oracle_prediction_filtered = oracle_prediction_filtered.to(device)
            oracle_prediction = oracle_prediction.to(device)

            y_pred = self.model(x)
            y_pred = torch.unsqueeze(y_pred, 1)
            y = torch.unsqueeze(y, 1)

            inf_classifier_pred = self.classif_inference.simple_inference(y_pred, filter_classes=True, softmax=False)
            
            if self.prop_logit != 100:
                cur_loss_mel = self.k_mel*self.loss_function_mel(y_pred,y)
                cur_loss_logit = self.k_logit*self.loss_function_logit(inf_classifier_pred, oracle_prediction)
                cur_loss = cur_loss_mel + cur_loss_logit
            else:
                cur_loss = self.loss_function_logit(inf_classifier_pred, oracle_prediction)


            losses_valid.append(cur_loss.detach())

            if self.prop_logit != 100:
                losses_mel_valid.append(cur_loss_mel.detach())
                losses_logit_valid.append(cur_loss_logit.detach())
                
        losses_valid = torch.Tensor(losses_valid)
        losses_mel_valid = torch.Tensor(losses_mel_valid)
        losses_logit_valid = torch.Tensor(losses_logit_valid)
        
        loss_valid = torch.mean(losses_valid)
        loss_mel_valid = torch.mean(losses_mel_valid)
        loss_logit_valid = torch.mean(losses_logit_valid)

        print(" => Validation loss at epoch {} is {:.4f}".format(cur_epoch+1, loss_valid))

        #save state dict if validation is better than best loss saved in the trainer
        if forced == True:
            self.best_loss = loss_valid
            self.best_state_dict = copy.deepcopy(self.model.state_dict())
            self.best_epoch = cur_epoch
        else:
            if label == 'VALIDATION':
                if loss_valid <= self.best_loss:
                    print('state dict saved at epoch ' + str(cur_epoch))
                    self.best_loss = loss_valid
                    self.best_state_dict = copy.deepcopy(self.model.state_dict())
                    self.best_epoch = cur_epoch
        
        return loss_valid.detach().cpu().numpy(), loss_mel_valid.detach().cpu().numpy(), loss_logit_valid.detach().cpu().numpy()
    
    def load_model(self, device):
        self.model = self.model.to(device)
        state_dict = torch.load(self.models_path / self.model_name, map_location=device)
        self.model.load_state_dict(state_dict)
        
    def save_model(self):
    
        """
        SAVE MODEL
        """
    
        torch.save(self.best_state_dict, self.models_path + self.model_name)
    
        """
        "SAVE MODEL SETTINGS"
        """
        
        transcoder = self.transcoder
        input_shape = self.model.input_shape
        output_shape = self.model.output_shape
        
        cnn_kernel_size = None
        cnn_dilation = None
        cnn_nb_layers = None
        cnn_nb_channels = None
        
        mlp_hl_1 = None
        mlp_hl_2 = None
        
        if 'cnn' in transcoder:
            cnn_kernel_size = self.model.kernel_size
            cnn_dilation = self.model.dilation
            cnn_nb_layers = self.model.nb_layers
            cnn_nb_channels = self.model.nb_channels
            
        if 'mlp' in transcoder:
            mlp_hl_1 = self.model.hl_1
            mlp_hl_2 = self.model.hl_2

        model_settings = {
            "model_type": transcoder,
          "input_shape": input_shape,
          "output_shape": output_shape,
          "cnn_kernel_size": cnn_kernel_size,
          "cnn_dilation": cnn_dilation,
          "cnn_nb_layers": cnn_nb_layers,
          "cnn_nb_channels": cnn_nb_channels,
          "mlp_hl_1": mlp_hl_1,
          "mlp_hl_2": mlp_hl_2,
          "mels_type": self.classifier,
          "batch_size": self.batch_size,
          "epochs": self.epochs,
          "settings": self.setting_data
        }
    
        with open(self.models_path + self.model_name + '_settings.yaml', 'w') as file:
            yaml.dump(model_settings, file)


#for CNN + PINV
class MelTrainer:
    def __init__(self, setting_data, model, models_path, transcoder, model_name, train_dataset=None, 
                 valid_dataset=None, eval_dataset=None, learning_rate=1e-3, dtype=torch.FloatTensor, classifier='PANN'):
        """
        Initializes the MelTrainer class. This class trains a model only on Mel spectrogram values.

        Args:
        - setting_data: The setting data for the model.
        - model: The model architecture.
        - models_path: The path to save the trained model.
        - transcoder: The transcoder type (cnn_pinv, mlp, mlp_pinv)
        - model_name: The name of the model.
        - train_dataset: The training dataset.
        - valid_dataset: The validation dataset.
        - eval_dataset: The evaluation dataset.
        - learning_rate: The learning rate for optimization.
        - dtype: The data type for the model (default: torch.FloatTensor).
        - classifier: The type of classifier (default: 'PANN').
        """
        self.setting_data = setting_data
        self.dtype = dtype
        self.classifier = classifier

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.eval_dataset = eval_dataset
        
        self.train_duration = 0

        for dataset in [train_dataset, valid_dataset, eval_dataset]:
            if dataset != None:
                self.flen = dataset.flen
                self.hlen = dataset.hlen
                self.sr = dataset.sr
                break
        
        self.model = model
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)
        
        self.models_path = models_path
        self.model_name = model_name
        self.transcoder = transcoder

        #for validation (not necessary if first validation is set before train)
        # self.best_loss = float('inf')
        # self.best_state_dict = copy.deepcopy(self.model.state_dict())
        # self.best_epoch = -1

        print('TRAINED MODEL')
        ut.count_parameters(self.model)
        
    def train(self, batch_size=64, epochs=10, device=torch.device("cpu")):
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        losses_train = []
        losses_valid = []
        losses_eval = []

        self.model.train()
        self.loss_function = nn.MSELoss()
        self.model = self.model.to(self.device)

        #fist validation
        loss_valid = self.validate(self.valid_dataset, 0, batch_size=batch_size, device=self.device, forced=True)
        losses_valid.append(loss_valid)
        
        loss_eval = self.validate(self.eval_dataset, 0, batch_size=batch_size, device=self.device, label='EVALUATION')
        losses_eval.append(loss_eval)
        
        #validation on evaluation
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False, drop_last=False)
        cur_loss = 0
        for cur_epoch in range(self.epochs):
            tqdm_it=tqdm(self.train_dataloader, 
                         desc='TRAINING: Epoch {}, Chunk {}/{}, loss: {:.4f}'
                         .format(cur_epoch+1, 0, 0, cur_loss))
            for (idx,x,y,_) in tqdm_it:

                start_time = time.time()

                x = x.type(self.dtype)
                y = y.type(self.dtype)
                
                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                y_pred = self.model(x)

                cur_loss = self.loss_function(y_pred,y)
                
                cur_loss.backward()

                batch_duration = time.time() - start_time
                self.train_duration += batch_duration
                
                self.optimizer.step()
                
                cur_loss = float(cur_loss.data)

                losses_train.append(cur_loss)
                
                tqdm_it.set_description('TRAINING: Epoch {}, Chunk {}/{}, loss: {:.4f}'
                                        .format(cur_epoch+1,0,0,cur_loss))                    
                
            #Validation
            loss_valid = self.validate(self.valid_dataset, cur_epoch, batch_size=batch_size, device=self.device)
            losses_valid.append(loss_valid)
            
            loss_eval = self.validate(self.eval_dataset, cur_epoch, batch_size=batch_size, device=self.device, label='EVALUATION')
            losses_eval.append(loss_eval)
        
        losses = {
                'losses_train': np.array(losses_train),
                'losses_valid': np.array(losses_valid),
                'losses_eval': np.array(losses_eval)
            }
        return(losses)
    
    def validate(self, dataset, cur_epoch, batch_size=64, device=torch.device("cpu"), label='VALIDATION', forced=False):
        self.model.eval
        losses_valid = []
        loss_function = nn.MSELoss()
        
        valid_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
        tqdm_it=tqdm(valid_dataloader, desc=label+': Chunk {}/{}'.format(0,0))

        for (idx,x,y,_) in tqdm_it:
            x = x.type(self.dtype)
            y = y.type(self.dtype)

            x = x.to(device)
            y = y.to(device)

            y_pred = self.model(x)

            cur_loss = loss_function(y_pred,y)

            losses_valid.append(cur_loss.detach())
                
        losses_valid = torch.Tensor(losses_valid)
        loss_valid = torch.mean(losses_valid)
        print(" => Validation loss at epoch {} is {:.4f}".format(cur_epoch+1, loss_valid))
        
        if forced == True:
            self.best_loss = loss_valid
            self.best_state_dict = copy.deepcopy(self.model.state_dict())
            self.best_epoch = cur_epoch
        else:
            if label == 'VALIDATION':
                if loss_valid <= self.best_loss:
                    self.best_loss = loss_valid
                    self.best_state_dict = copy.deepcopy(self.model.state_dict())
                    self.best_epoch = cur_epoch
        
        return loss_valid.detach().cpu().numpy()
    
    def load_model(self, device):
        self.model = self.model.to(device)
        state_dict = torch.load(self.models_path / self.model_name, map_location=device)
        self.model.load_state_dict(state_dict)
        
    def save_model(self):
    
        """
        SAVE MODEL
        """
    
        torch.save(self.best_state_dict, self.models_path + self.model_name)
    
        """
        "SAVE MODEL SETTINGS"
        """
        
        transcoder = self.transcoder
        input_shape = self.model.input_shape
        output_shape = self.model.output_shape
        
        cnn_kernel_size = None
        cnn_dilation = None
        cnn_nb_layers = None
        cnn_nb_channels = None
        
        mlp_hl_1 = None
        mlp_hl_2 = None
        
        if 'cnn' in transcoder:
            cnn_kernel_size = self.model.kernel_size
            cnn_dilation = self.model.dilation
            cnn_nb_layers = self.model.nb_layers
            cnn_nb_channels = self.model.nb_channels
            
        if 'mlp' in transcoder:
            mlp_hl_1 = self.model.hl_1
            mlp_hl_2 = self.model.hl_2

        model_settings = {
            "model_type": transcoder,
          "input_shape": input_shape,
          "output_shape": output_shape,
          "cnn_kernel_size": cnn_kernel_size,
          "cnn_dilation": cnn_dilation,
          "cnn_nb_layers": cnn_nb_layers,
          "cnn_nb_channels": cnn_nb_channels,
          "mlp_hl_1": mlp_hl_1,
          "mlp_hl_2": mlp_hl_2,
          "mels_type": self.classifier,
          "batch_size": self.batch_size,
          "epochs": self.epochs,
          "settings": self.setting_data
        }

        with open(self.models_path + self.model_name + '_settings.yaml', 'w') as file:
            yaml.dump(model_settings, file)

#for CNN + PINV
class LogitTrainer:
    def __init__(self, setting_data, model, models_path, transcoder, model_name, train_dataset=None, 
                 valid_dataset=None, eval_dataset=None, learning_rate=1e-3, dtype=torch.FloatTensor, 
                 classifier='PANN'):
        """
        Initializes a LogitTrainer object. The LogitTrainer trains a model only on logits values.

        Args:
            setting_data (dict): The setting data for the model.
            model: The neural network model.
            models_path (str): The path to save the trained models.
            transcoder (str): The transcoder type (effnet_b0, effnet_b7, self)
            model_name (str): The name of the model.
            train_dataset (Dataset): The training dataset.
            valid_dataset (Dataset): The validation dataset.
            eval_dataset (Dataset): The evaluation dataset.
            learning_rate (float): The learning rate for the optimizer.
            dtype (torch.dtype): The data type for the tensors.
            classifier (str): The type of classifier ("PANN" or "YamNet")
        """
        #factors in front of loss functions
        
        self.setting_data = setting_data
        self.dtype = dtype
        self.classifier = classifier
        
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.eval_dataset = eval_dataset

        self.train_duration = 0

        for dataset in [train_dataset, valid_dataset, eval_dataset]:
            if dataset != None:
                self.flen = dataset.flen
                self.hlen = dataset.hlen
                self.sr = dataset.sr
                break
        
        self.model = model
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)
        
        self.models_path = models_path
        self.model_name = model_name
        self.transcoder = transcoder

        #for validation (not necessary if first validation is set before train)
        self.best_loss = float('inf')
        self.best_state_dict = copy.deepcopy(self.model.state_dict())
        self.best_epoch = -1

        print('TRAINED MODEL')
        ut.count_parameters(self.model)
        
    def train(self, batch_size=64, epochs=10, device=torch.device("cpu")):
        
        mels_type = self.eval_dataset.mels_type
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device

        losses_train = []
        losses_valid = []
        losses_eval = []

        self.model.train()
        self.loss_function_mel = nn.MSELoss()
        self.loss_function_logit = nn.BCELoss()

        self.model = self.model.to(self.device)

        #fist validation
        loss_valid = self.validate(self.valid_dataset, 0, batch_size=batch_size, device=self.device, forced=True)
        losses_valid.append(loss_valid)
        
        loss_eval = self.validate(self.eval_dataset, 0, batch_size=batch_size, device=self.device, label='EVALUATION')
        losses_eval.append(loss_eval)
        
        #validation on evaluation
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False, drop_last=False)
        cur_loss = 0

        #callback = ParameterCountCallback(self.model)

        for cur_epoch in range(self.epochs):
            tqdm_it=tqdm(self.train_dataloader, 
                         desc='TRAINING: Epoch {}, Chunk {}/{}, loss: {:.4f}'
                         .format(cur_epoch+1, 0, 0, cur_loss))
            for (idx, x, y, input_fname, oracle_prediction, oracle_prediction_filtered) in tqdm_it:
                
                start_time = time.time()

                x = x.type(self.dtype)
                y = y.type(self.dtype)
                oracle_prediction_filtered = oracle_prediction_filtered.type(self.dtype)
                oracle_prediction = oracle_prediction.type(self.dtype)

                x = x.to(self.device)
                y = y.to(self.device)


                oracle_prediction_filtered = oracle_prediction_filtered.to(self.device)
                oracle_prediction = oracle_prediction.to(self.device)

                self.optimizer.zero_grad()
                
                inf_classifier_pred = self.model(x)

                cur_loss = self.loss_function_logit(inf_classifier_pred, oracle_prediction)

                cur_loss.backward()

                batch_duration = time.time() - start_time
                self.train_duration += batch_duration
                
                # duration = time.time() - start_time
                # print(f'duration backward: {duration}')

                # start_time = time.time()

                self.optimizer.step()
                
                cur_loss = float(cur_loss.data)

                losses_train.append(cur_loss)

                tqdm_it.set_description('TRAINING: Epoch {}, Chunk {}/{}, loss: {:.4f}'
                                        .format(cur_epoch + 1, 0, 0, cur_loss))

                
            #Validation
            loss_valid = self.validate(self.valid_dataset, cur_epoch, batch_size=batch_size, device=self.device)
            losses_valid.append(loss_valid)
            
            loss_eval = self.validate(self.eval_dataset, cur_epoch, batch_size=batch_size, device=self.device, label='EVALUATION')
            losses_eval.append(loss_eval)
        
        losses = {
                'losses_train': np.array(losses_train),
                'losses_valid': np.array(losses_valid),
                'losses_eval': np.array(losses_eval)
            }

        
        return(losses)
    
    def validate(self, dataset, cur_epoch, batch_size=64, device=torch.device("cpu"), label='VALIDATION', forced=False):
        self.model.eval
        loss_valid = 0
        
        losses_valid = []

        loss_function_logit = nn.BCELoss()

        valid_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
        tqdm_it=tqdm(valid_dataloader, desc=label+': Chunk {}/{}'.format(0,0))
        for (idx, x , y, input_fname, oracle_prediction, oracle_prediction_filtered) in tqdm_it:
            x = x.type(self.dtype)
            y = y.type(self.dtype)
            oracle_prediction_filtered = oracle_prediction_filtered.type(self.dtype)
            oracle_prediction = oracle_prediction.type(self.dtype)

            x = x.to(device)
            y = y.to(device)
            oracle_prediction_filtered = oracle_prediction_filtered.to(device)
            oracle_prediction = oracle_prediction.to(device)

            inf_classifier_pred = self.model(x)

            cur_loss = loss_function_logit(inf_classifier_pred, oracle_prediction)

            losses_valid.append(cur_loss.detach())
                
        losses_valid = torch.Tensor(losses_valid)
        loss_valid = torch.mean(losses_valid)
        print(" => Validation loss at epoch {} is {:.4f}".format(cur_epoch+1, loss_valid))
        
        if forced == True:
            self.best_loss = loss_valid
            self.best_state_dict = copy.deepcopy(self.model.state_dict())
            self.best_epoch = cur_epoch
        else:
            if label == 'VALIDATION':
                if loss_valid <= self.best_loss:
                    print('state dict saved at epoch ' + str(cur_epoch))
                    self.best_loss = loss_valid
                    self.best_state_dict = copy.deepcopy(self.model.state_dict())
                    self.best_epoch = cur_epoch
        
        return loss_valid.detach().cpu().numpy()
    
    def load_model(self, device):
        self.model = self.model.to(device)
        state_dict = torch.load(self.models_path / self.model_name, map_location=device)
        self.model.load_state_dict(state_dict)
        
    def save_model(self):
    
        """
        SAVE MODEL
        """
    
        torch.save(self.best_state_dict, self.models_path + self.model_name)
    
        """
        "SAVE MODEL SETTINGS"
        """
        
        transcoder = self.transcoder
        
        cnn_kernel_size = None
        cnn_dilation = None
        cnn_nb_layers = None
        cnn_nb_channels = None

        input_shape = None
        output_shape = None

        mlp_hl_1 = None
        mlp_hl_2 = None
        
        if transcoder not in ["effnet_b0", "effnet_b7"]:
            input_shape = self.model.input_shape
            output_shape = self.model.output_shape

        if 'cnn' in transcoder:
            cnn_kernel_size = self.model.kernel_size
            cnn_dilation = self.model.dilation
            cnn_nb_layers = self.model.nb_layers
            cnn_nb_channels = self.model.nb_channels
            
        if 'mlp' in transcoder:
            mlp_hl_1 = self.model.hl_1
            mlp_hl_2 = self.model.hl_2

        model_settings = {
            "model_type": transcoder,
          "input_shape": input_shape,
          "output_shape": output_shape,
          "cnn_kernel_size": cnn_kernel_size,
          "cnn_dilation": cnn_dilation,
          "cnn_nb_layers": cnn_nb_layers,
          "cnn_nb_channels": cnn_nb_channels,
          "mlp_hl_1": mlp_hl_1,
          "mlp_hl_2": mlp_hl_2,
          "mels_type": self.classifier,
          "batch_size": self.batch_size,
          "epochs": self.epochs,
          "settings": self.setting_data
        }
    
        with open(self.models_path + self.model_name + '_settings.yaml', 'w') as file:
            yaml.dump(model_settings, file)
