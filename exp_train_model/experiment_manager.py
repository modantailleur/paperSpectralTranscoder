#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:47:28 2022

@author: user
"""

import torch
import torch.nn as nn
from pathlib import Path
import models as md
from data_loaders import MelDataset, OutputsDataset, MelLogitDataset
from trainers import HybridTrainer, MelTrainer, LogitTrainer
from evaluaters import DLTranscoderEvaluater, PinvEvaluater, OracleEvaluater, TSEvaluater
import utils.util as ut
import numpy as np
import yaml
from tqdm import tqdm
from pann.pann_mel_inference import PannMelInference
from yamnet.yamnet_mel_inference import YamnetMelInference
import torch.nn.functional as F

def train_dl_model(setting_exp):
    """
    Trains a deep learning model based on the provided SettingExp experimental settings. 
    Different trainers are used depending of the deep learning model:
    - MelTrainer for transcoders that are trained directly by comparing the groundtruth and predicted Mel spectrogram
    - HybridTrainer for transcoders that are trained by comparing the groundtruth logits of the pre-trained models with 
        the logits of PANN that takes as input a predicted Mel spectrogram. The trainer is hybrid, which means 
        that it can also be trained partly on groundtruth Mel spectrograms if prop_logit is different than None
    - LogitTrainer for non-transcoding methods, that only involve training on logits

    The models are saved during the training, in the path corresponding to the "model_path" attribute of setting_exp

    Args:
        setting_exp (object): A SettingExp instance from the exp_train_model/main_doce_training.py file, containing the experimental settings.

    Returns:
        losses (dict), duration (float): A dict containing the losses during training, and the duration of training (in seconds).
    """
    if setting_exp.ts == 0:
        # WITHOUT TS
        train_dataset = MelDataset(setting_exp.setting_data, subset='train', classifier=setting_exp.classifier, project_data_path=setting_exp.project_data_path)
        valid_dataset = MelDataset(setting_exp.setting_data, subset='valid', classifier=setting_exp.classifier, project_data_path=setting_exp.project_data_path)
        eval_dataset = MelDataset(setting_exp.setting_data, subset='eval', classifier=setting_exp.classifier, project_data_path=setting_exp.project_data_path)
    else:
        # WITH TS
        train_dataset = MelLogitDataset(setting_exp.setting_data, subset='train', classifier=setting_exp.classifier, project_data_path=setting_exp.project_data_path, outputs_oracle_path=setting_exp.outputs_oracle_path)
        valid_dataset = MelLogitDataset(setting_exp.setting_data, subset='valid', classifier=setting_exp.classifier, project_data_path=setting_exp.project_data_path, outputs_oracle_path=setting_exp.outputs_oracle_path)
        eval_dataset = MelLogitDataset(setting_exp.setting_data, subset='eval', classifier=setting_exp.classifier, project_data_path=setting_exp.project_data_path, outputs_oracle_path=setting_exp.outputs_oracle_path)

    #used as a reference to set the models inputs/outputs to the correct shape
    ref_dataset = train_dataset
    
    #get correct transforms depending on the target mel spectrogram representation
    #and the third octave band spectrogram representation of the data to transcode
    tho_tr, mels_tr = ut.get_transforms(sr=ref_dataset.sr, 
                                          flen=ref_dataset.flen,
                                          hlen=ref_dataset.hlen,
                                          classifier=setting_exp.classifier,
                                          device=setting_exp.device, tho_freq=setting_exp.tho_freq, tho_time=setting_exp.tho_time, mel_template=setting_exp.mel_template)
        
    if setting_exp.transcoder == 'mlp':
        input_shape = (ref_dataset.n_tho_frames_per_file, ref_dataset.n_tho)
        output_shape = (ref_dataset.n_mel_frames_per_file, ref_dataset.n_mels)
        model = md.MLP(input_shape, output_shape, hl_1=setting_exp.hidden1, hl_2=setting_exp.hidden2)
        
    if setting_exp.transcoder == 'mlp_pinv':
        input_shape = (ref_dataset.n_tho_frames_per_file, ref_dataset.n_mels)
        output_shape = (ref_dataset.n_mel_frames_per_file, ref_dataset.n_mels)
        model = md.MLPPINV(input_shape, output_shape, tho_tr, mels_tr, hl_1=setting_exp.hidden1, hl_2=setting_exp.hidden2, device=setting_exp.device)
    
    if setting_exp.transcoder == 'cnn_pinv':
        input_shape = (ref_dataset.n_tho_frames_per_file, ref_dataset.n_mels)
        output_shape = (ref_dataset.n_mel_frames_per_file, ref_dataset.n_mels)
        model = md.CNN(input_shape=input_shape, output_shape=output_shape, tho_tr=tho_tr,
                mels_tr=mels_tr, kernel_size=setting_exp.kernel_size, dilation=setting_exp.dilation, 
                nb_layers=setting_exp.nb_layers,nb_channels=setting_exp.nb_channels, device=setting_exp.device, input_is_db=setting_exp.input_is_db)
    
    if setting_exp.transcoder == 'self':
        input_shape = (ref_dataset.n_tho_frames_per_file, ref_dataset.n_mels)
        output_shape = (ref_dataset.n_mel_frames_per_file, ref_dataset.n_mels)
        if setting_exp.classifier == 'PANN':
            model = md.PANNPINV(input_shape=input_shape, output_shape=output_shape, tho_tr=tho_tr, mels_tr=mels_tr, dtype=setting_exp.dtype, device=setting_exp.device)
        if setting_exp.classifier == 'YamNet':
            model = md.YAMNETPINV(input_shape=input_shape, output_shape=output_shape, tho_tr=tho_tr, mels_tr=mels_tr, dtype=setting_exp.dtype, device=setting_exp.device)
    
    if setting_exp.transcoder in ['effnet_b0', 'effnet_b7']:
        input_shape = (ref_dataset.n_tho_frames_per_file, ref_dataset.n_mels)
        output_shape = (ref_dataset.n_mel_frames_per_file, ref_dataset.n_mels)
        model = md.EffNet(effnet_type=setting_exp.transcoder, mels_tr=mels_tr, dtype=setting_exp.dtype, device=setting_exp.device)

    if setting_exp.prop_logit is None:
        if setting_exp.transcoder in ['self', 'effnet_b0', 'effnet_b7']:
            trainer = LogitTrainer(setting_exp.setting_data, model, setting_exp.model_path, setting_exp.transcoder, setting_exp.model_name, 
                                train_dataset=train_dataset, 
                                valid_dataset=valid_dataset, 
                                eval_dataset=eval_dataset,
                                learning_rate=setting_exp.learning_rate,
                                classifier=setting_exp.classifier)
        else:
            trainer = MelTrainer(setting_exp.setting_data, model, setting_exp.model_path, setting_exp.transcoder, setting_exp.model_name, 
                                    train_dataset=train_dataset, 
                                    valid_dataset=valid_dataset, 
                                    eval_dataset=eval_dataset,
                                    learning_rate=setting_exp.learning_rate,
                                    classifier=setting_exp.classifier)

    else:
        trainer = HybridTrainer(setting_exp.setting_data, model, setting_exp.model_path, setting_exp.transcoder, setting_exp.model_name, 
                                train_dataset=train_dataset, 
                                valid_dataset=valid_dataset, 
                                eval_dataset=eval_dataset,
                                learning_rate=setting_exp.learning_rate,
                                classifier=setting_exp.classifier, prop_logit=setting_exp.prop_logit)

    
    losses = trainer.train(batch_size=setting_exp.batch_size, epochs=setting_exp.epoch, device=setting_exp.device)

    trainer.save_model()
    
    return(losses, trainer.train_duration)   

def evaluate_dl_model(setting_exp):
    """
    Evaluates a deep learning model based on the provided SettingExp experimental settings. 
    Different evaluaters are used depending of the trained deep learning model:
    - DLTranscoderEvaluater for transcoders 
    - TSEvaluater for non-transcoding models (self and efficient nets)

    As the Mel data stored during this evaluation might be too heavy in memory, the results
    of the evaluation both in term of Mel data and logit data are stored directly inside the evaluaters, 
    and not in the exp_train_model/main_doce_training.py file.

    Args:
        setting_exp (object): A SettingExp instance from the exp_train_model/main_doce_training.py file, containing the experimental settings.
    """

    with open(setting_exp.model_path + setting_exp.model_name + '_settings.yaml') as file:
        setting_model = yaml.load(file, Loader=yaml.FullLoader)
    
    input_shape = setting_model.get('input_shape')
    output_shape = setting_model.get('output_shape')
    
    cnn_kernel_size = setting_model.get('cnn_kernel_size')
    cnn_dilation = setting_model.get('cnn_dilation')
    cnn_nb_layers = setting_model.get('cnn_nb_layers')
    cnn_nb_channels = setting_model.get('cnn_nb_channels')

    mlp_hl_1 = setting_model.get('mlp_hl_1')
    mlp_hl_2 = setting_model.get('mlp_hl_2')
    
    #from settings of the model
    classifier = setting_model.get('mels_type')
    
    eval_dataset = MelDataset(setting_exp.setting_data, subset='eval', classifier=classifier, project_data_path=setting_exp.project_data_path)
    ref_dataset = eval_dataset
    
    #handle models that have a pinv in them
    if setting_exp.transcoder in ['mlp_pinv', 'cnn_pinv', 'mlp', 'self', 'effnet_b0', 'effnet_b7']:
        tho_tr, mels_tr = ut.get_transforms(sr=ref_dataset.sr, 
                                          flen=ref_dataset.flen,
                                          hlen=ref_dataset.hlen,
                                          classifier=classifier,
                                          device=setting_exp.device, tho_freq=setting_exp.tho_freq, tho_time=setting_exp.tho_time, mel_template=setting_exp.mel_template)
        
        
    if setting_exp.transcoder == 'mlp':
        model = md.MLP(input_shape, output_shape, hl_1=mlp_hl_1, hl_2=mlp_hl_2)
        
    if setting_exp.transcoder == 'mlp_pinv':
        model = md.MLPPINV(input_shape, output_shape, tho_tr, mels_tr, hl_1=mlp_hl_1, hl_2=mlp_hl_2, device=setting_exp.device,
                            input_is_db=setting_exp.input_is_db)
    
    if setting_exp.transcoder == 'cnn_pinv':
        model = md.CNN(input_shape=input_shape, output_shape=output_shape, tho_tr=tho_tr, 
                        mels_tr=mels_tr, kernel_size=cnn_kernel_size, dilation=cnn_dilation, 
                        nb_layers=cnn_nb_layers, nb_channels=cnn_nb_channels, device=setting_exp.device,
                        input_is_db=setting_exp.input_is_db)
        
    if setting_exp.transcoder == 'self':
        if mels_tr.name == 'PANN':
            model = md.PANNPINV(input_shape=input_shape, output_shape=output_shape, tho_tr=tho_tr, mels_tr=mels_tr, dtype=setting_exp.dtype, device=setting_exp.device)
        if mels_tr.name == 'YamNet':
            model = md.YAMNETPINV(input_shape=input_shape, output_shape=output_shape, tho_tr=tho_tr, mels_tr=mels_tr, dtype=setting_exp.dtype, device=setting_exp.device)
    
    if setting_exp.transcoder in ['effnet_b0', 'effnet_b7']:
        model = md.EffNet(mels_tr=mels_tr, effnet_type=setting_exp.transcoder, dtype=setting_exp.dtype, device=setting_exp.device)

    if setting_exp.transcoder not in ['effnet_b0', 'effnet_b7', 'self']:
        evaluater = DLTranscoderEvaluater(setting_exp.setting_data, model, setting_exp.model_path, setting_exp.model_name, 
                                    setting_exp.outputs_path,
                                eval_dataset=eval_dataset)
    else:
        evaluater = TSEvaluater(setting_exp.setting_data, model, setting_exp.model_path, setting_exp.model_name, 
                            setting_exp.outputs_path,
                        eval_dataset=eval_dataset)
    
    evaluater.load_model(device=setting_exp.device)
    evaluater.evaluate(batch_size=setting_exp.batch_size, device=setting_exp.device)
    return()

def evaluate_pinv(setting_exp):
    """
    Evaluates the PINV transcoder based on the provided SettingExp experimental settings. This function doesn't return anything, 
    as logits and Mel spectrograms are saved in .dat format in the evaluaters to make 
    sure that memory isn't overloaded.

    Args:
        setting_exp (object): A SettingExp instance from the exp_train_model/main_doce_training.py file, containing the experimental settings.
    """
    
    eval_dataset = MelDataset(setting_exp.setting_data, subset='eval', classifier=setting_exp.classifier, project_data_path=setting_exp.project_data_path)
    ref_dataset = eval_dataset
    
    tho_tr, mels_tr = ut.get_transforms(sr=ref_dataset.sr, 
                                      flen=ref_dataset.flen,
                                      hlen=ref_dataset.hlen,
                                      classifier=setting_exp.classifier,
                                      device=setting_exp.device, tho_freq=setting_exp.tho_freq, tho_time=setting_exp.tho_time, mel_template=setting_exp.mel_template)
    
    evaluator = PinvEvaluater(eval_dataset, tho_tr, mels_tr, 
                              setting_exp.outputs_path)
    
    evaluator.evaluate(batch_size=setting_exp.batch_size, device=setting_exp.device)
    return()

    
def evaluate_oracle(setting_exp):
    """
    Evaluates the oracle transcoder based on the provided SettingExp experimental settings.
    The oracle is the model that takes perfectly predictions of Mel spectrograms as input.
    This evaluation function is just a check to see if any mistakes could have been made
    at any stage of the calculation.

    The oracle results are also used for training the deep learning models, using the logits
    and Mel spectrograms saved in this function. This function doesn't return anything, 
    as logits and Mel spectrograms are saved in .dat format in the evaluaters to make 
    sure that memory isn't overloaded.

    Args:
        setting_exp (object): A SettingExp instance from the exp_train_model/main_doce_training.py file, containing the experimental settings.
    """
    eval_dataset = MelDataset(setting_exp.setting_data, subset='eval', classifier=setting_exp.classifier, project_data_path=setting_exp.project_data_path)
    valid_dataset = MelDataset(setting_exp.setting_data, subset='valid', classifier=setting_exp.classifier, project_data_path=setting_exp.project_data_path)
    train_dataset = MelDataset(setting_exp.setting_data, subset='train', classifier=setting_exp.classifier, project_data_path=setting_exp.project_data_path)    
    
    evaluator = OracleEvaluater(eval_dataset, 
                                setting_exp.outputs_path)
    
    evaluator.evaluate(batch_size=setting_exp.batch_size, device=setting_exp.device)

    evaluator = OracleEvaluater(eval_dataset, 
                            setting_exp.outputs_path, label='eval', tvb=True)

    evaluator.evaluate(batch_size=setting_exp.batch_size, device=setting_exp.device)

    evaluator = OracleEvaluater(valid_dataset, 
                            setting_exp.outputs_path, label='valid', tvb=True)

    evaluator.evaluate(batch_size=setting_exp.batch_size, device=setting_exp.device)

    evaluator = OracleEvaluater(train_dataset, 
                            setting_exp.outputs_path, label='train', tvb=True)

    evaluator.evaluate(batch_size=setting_exp.batch_size, device=setting_exp.device)

    return()

def compute_metrics(setting_exp):
    """
    Calculate evaluation metrics for a classifier model.

    Args:
        setting_exp (object): A SettingExp instance from the exp_train_model/main_doce_training.py file, containing the experimental settings.
 
    Returns:
        metrics: A dictionary containing the calculated metrics.
        others: A dictionary containing additional information such as predicted and grountruth labels.

    """
    #most dominent class in strings
    if setting_exp.classifier == 'PANN':
        classif_inference = PannMelInference(device=setting_exp.device)
    if setting_exp.classifier == 'YamNet':
        classif_inference = YamnetMelInference(device=setting_exp.device)

    outputs_dataset = OutputsDataset(setting_data=setting_exp.setting_data, 
                                     outputs_path=setting_exp.outputs_path, outputs_oracle_path=setting_exp.outputs_oracle_path, no_mels=setting_exp.no_mels,
                                     classifier=setting_exp.classifier, project_data_path=setting_exp.project_data_path)
    
    #losses outputted by the function
    if not setting_exp.no_mels:
        mel_mse = torch.Tensor([]).to(setting_exp.device)
    else:
        mel_mse = None

    logit_mse = torch.Tensor([]).to(setting_exp.device)
    logit_tvb_mse = torch.Tensor([]).to(setting_exp.device)

    logit_bce = torch.Tensor([]).to(setting_exp.device)
    logit_tvb_bce = torch.Tensor([]).to(setting_exp.device)
    
    logit_cos = torch.Tensor([]).to(setting_exp.device)
    logit_tvb_cos = torch.Tensor([]).to(setting_exp.device)

    logit_kl = torch.Tensor([]).to(setting_exp.device)
    logit_tvb_kl = torch.Tensor([]).to(setting_exp.device)

    labels_tvb = np.array([])

    labels = np.array([])

    lf_mels_mse = nn.MSELoss(reduction='none')
    lf_classifier_mse = nn.MSELoss(reduction='none')
    lf_classifier_bce =  nn.BCELoss(reduction='none')
    lf_cos = nn.CosineSimilarity(dim=2, eps=1e-08)
    lf_kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
    
    logit_dataloader = torch.utils.data.DataLoader(outputs_dataset, batch_size=setting_exp.batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
    tqdm_it=tqdm(logit_dataloader, desc='METRICS: Chunk {}/{}'.format(0,0))

    sum_oracle_fc = torch.zeros(classif_inference.n_labels)
    sum_oracle_fc_tvb = torch.zeros(classif_inference.n_labels_tvb)
    sum_model_fc = torch.zeros(classif_inference.n_labels)
    sum_model_fc_tvb = torch.zeros(classif_inference.n_labels_tvb)

    #loop on transcoder's output (model_mel, model_logit) and classifier's outputs (oracle_mel, oracle_logit) on 10s audio files 
    # to calculate metrics at the scale of the audio file (important for significance calculation)
    for (oracle_mel, model_mel, oracle_logit, model_logit) in tqdm_it:
        
        #set dtype and put on device
        if not setting_exp.no_mels:
            oracle_mel = oracle_mel.type(setting_exp.dtype)
            model_mel = model_mel.type(setting_exp.dtype)

            oracle_mel = oracle_mel.to(setting_exp.device)
            model_mel = model_mel.to(setting_exp.device)
        
        oracle_logit = oracle_logit.type(setting_exp.dtype)
        model_logit = model_logit.type(setting_exp.dtype)

        oracle_logit = oracle_logit.to(setting_exp.device)        
        model_logit = model_logit.to(setting_exp.device)

        #calculate metric only with tvb (traffic, voices, birds) classes present in file utils/sub_classes.xlsx
        oracle_logit_tvb = classif_inference.logit_to_logit_tvb(oracle_logit)
        model_logit_tvb = classif_inference.logit_to_logit_tvb(model_logit)

        #calculate the index of the class that has the maximum prediction value
        argmax_oracle_logit = torch.argmax(oracle_logit.mean(dim=1), dim=1)
        argmax_oracle_logit_tvb = torch.argmax(oracle_logit_tvb.mean(dim=1), dim=1)
        argmax_model_logit = torch.argmax(model_logit.mean(dim=1), dim=1)
        argmax_model_logit_tvb = torch.argmax(model_logit_tvb.mean(dim=1), dim=1)
        
        #convert the previous value into a one hot vector 
        oracle_logit_hot = F.one_hot(argmax_oracle_logit.to(torch.int64), num_classes=classif_inference.n_labels)
        oracle_logit_tvb_hot = F.one_hot(argmax_oracle_logit_tvb.to(torch.int64), num_classes=classif_inference.n_labels_tvb)
        model_logit_hot = F.one_hot(argmax_model_logit.to(torch.int64), num_classes=classif_inference.n_labels)
        model_logit_tvb_hot = F.one_hot(argmax_model_logit_tvb.to(torch.int64), num_classes=classif_inference.n_labels_tvb)

        # check if oracle and logit have the same first class prediction
        eq_hot = torch.logical_and(oracle_logit_hot, model_logit_hot)
        eq_tvb_hot = torch.logical_and(oracle_logit_tvb_hot, model_logit_tvb_hot)

        # calculate the number of files that has their first class in the different classes
        sum_oracle_fc = sum_oracle_fc + oracle_logit_hot.sum(dim=0).detach().cpu()
        sum_oracle_fc_tvb = sum_oracle_fc_tvb + oracle_logit_tvb_hot.sum(dim=0).detach().cpu()

        # calculate the number of files that have been rightfully predicted in each class
        sum_model_fc = sum_model_fc + eq_hot.sum(dim=0).detach().cpu()
        sum_model_fc_tvb = sum_model_fc_tvb + eq_tvb_hot.sum(dim=0).detach().cpu()

        # predicted classes as str
        batch_labels_tvb = classif_inference.logit_to_labels(model_logit_tvb.detach().cpu(), tvb=True)
        batch_labels = classif_inference.logit_to_labels(model_logit.detach().cpu())

        labels_tvb = np.concatenate((labels_tvb, batch_labels_tvb), axis=0)
        labels = np.concatenate((labels, batch_labels), axis=0)
        
        # mse calculate on mels
        if not setting_exp.no_mels:
            batch_mel_mse =  lf_mels_mse(model_mel, oracle_mel).mean(dim=(1,2,3)) 
            mel_mse = torch.cat((mel_mse,batch_mel_mse))

        # mse calculated on logits
        batch_logit_mse = lf_classifier_mse(model_logit, oracle_logit).mean(dim=(1,2))    
        logit_mse = torch.cat((logit_mse,batch_logit_mse))

        batch_logit_tvb_mse = lf_classifier_mse(model_logit_tvb, oracle_logit_tvb).mean(dim=(1,2))        
        logit_tvb_mse = torch.cat((logit_tvb_mse,batch_logit_tvb_mse))

        # bce calculated on logits
        batch_logit_bce = lf_classifier_bce(model_logit, oracle_logit).mean(dim=(1,2))  
        logit_bce = torch.cat((logit_bce,batch_logit_bce))

        batch_logit_tvb_bce = lf_classifier_bce(model_logit_tvb, oracle_logit_tvb).mean(dim=(1,2))    
        logit_tvb_bce = torch.cat((logit_tvb_bce,batch_logit_tvb_bce))

        # cos similarity calculated on logits
        batch_logit_cos = lf_cos(model_logit, oracle_logit).mean(dim=1)
        logit_cos = torch.cat((logit_cos, batch_logit_cos))

        batch_logit_tvb_cos = lf_cos(model_logit_tvb, oracle_logit_tvb).mean(dim=1)
        logit_tvb_cos = torch.cat((logit_tvb_cos, batch_logit_tvb_cos))

        #KL div calculated on logits
        #new calculation for KL: have to calculate it like that because "batchmean" is
        #recommanded by pytorch documentation to calculate the real KL divergence. Can't
        #calculate it simply like the BCE loss. 
        epsilon = 1e-10
        model_logit_avg = torch.mean(model_logit, dim=1)
        oracle_logit_avg = torch.mean(oracle_logit, dim=1)
        L = []
        for k in range(model_logit.shape[0]):
            L.append(lf_kl(torch.log(model_logit_avg[k]+epsilon), torch.log(oracle_logit_avg[k]+epsilon)))
        batch_logit_kl = torch.Tensor(L).to(setting_exp.device)

        logit_kl = torch.cat((logit_kl, batch_logit_kl))

        model_logit_tvb_avg = torch.mean(model_logit_tvb, dim=1)
        oracle_logit_tvb_avg = torch.mean(oracle_logit_tvb, dim=1)
        L = []
        for k in range(model_logit.shape[0]):
            L.append(lf_kl(torch.log(model_logit_tvb_avg[k]+epsilon), torch.log(oracle_logit_tvb_avg[k]+epsilon)))
        batch_logit_tvb_kl = torch.Tensor(L).to(setting_exp.device)
        logit_tvb_kl = torch.cat((logit_tvb_kl, batch_logit_tvb_kl))

    # calculate the weighted mean top1 of the accuracy accross each class (ptopafc weighted) --> accuracy weighted by class 
    # only get the indexes of classes that have at least been predicted once as first class
    nonzero_oracle_fc=torch.nonzero(sum_oracle_fc)
    ptopafc_weighted = (sum_model_fc[nonzero_oracle_fc] / sum_oracle_fc[nonzero_oracle_fc]).mean()

    nonzero_oracle_fc_tvb=torch.nonzero(sum_oracle_fc_tvb)
    ptopafc_weighted_tvb = (sum_model_fc_tvb[nonzero_oracle_fc_tvb] / sum_oracle_fc_tvb[nonzero_oracle_fc_tvb]).mean()

    # calculate the mean top1 of the accuracy accross each class (ptopafc)--> overall accuracy
    ptopafc = sum_model_fc[nonzero_oracle_fc].sum() / sum_oracle_fc[nonzero_oracle_fc].sum()
    ptopafc_tvb = sum_model_fc_tvb[nonzero_oracle_fc_tvb].sum() / sum_oracle_fc_tvb[nonzero_oracle_fc_tvb].sum()

    metrics = {
        'logit_mse': logit_mse.detach().cpu().numpy(),
        'logit_bce':logit_bce.detach().cpu().numpy(),
        'logit_tvb_mse':logit_tvb_mse.detach().cpu().numpy(),
        'logit_tvb_bce': logit_tvb_bce.detach().cpu().numpy(),

        'logit_cos': logit_cos.detach().cpu().numpy(),
        'logit_tvb_cos': logit_tvb_cos.detach().cpu().numpy(),

        'logit_kl': logit_kl.detach().cpu().numpy(),
        'logit_tvb_kl': logit_tvb_kl.detach().cpu().numpy(),

        'ptopafc': ptopafc.detach().cpu().numpy(),
        'ptopafc_tvb': ptopafc_tvb.detach().cpu().numpy(),
        'ptopafc_weighted' : ptopafc_weighted.detach().cpu().numpy(),
        'ptopafc_weighted_tvb': ptopafc_weighted_tvb.detach().cpu().numpy()
    }

    others = {
        'labels' : labels,
        'labels_tvb': labels_tvb
    }

    #only add the mel MSE metric if the trained model uses mels (hence if it is a transcoder)
    if not setting_exp.no_mels:
        metrics['mel_mse'] = mel_mse.detach().cpu().numpy()

    # just used for pretty display in terminal
    p_logit_mse = float("{:.8f}".format(torch.mean(logit_mse)))
    p_logit_bce = float("{:.8f}".format(torch.mean(logit_bce)))
    p_logit_kl = float("{:.8f}".format(torch.mean(logit_kl)))
    p_ptopafc = float("{:.8f}".format(torch.mean(ptopafc)))
    p_ptopafc_tvb = float("{:.8f}".format(torch.mean(ptopafc_tvb)))
    if not setting_exp.no_mels:
        p_mel_mse = float("{:.8f}".format(torch.mean(mel_mse)))

    print('=> METRICS: evaluation')
    print(f'logit_mse: {p_logit_mse}')
    print(f'logit_bce: {p_logit_bce}')
    print(f'logit_kl: {p_logit_kl}')
    print(f'ptopafc: {p_ptopafc}')
    print(f'ptopafc_tvb: {p_ptopafc_tvb}')
    print('random logit:' + str(1/torch.numel(nonzero_oracle_fc)))
    print('random logit tvb:' + str(1/torch.numel(nonzero_oracle_fc_tvb)))
    if not setting_exp.no_mels:
        print(f'mel_mse: {p_mel_mse}')

    return(metrics, others)
    
