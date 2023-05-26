from textwrap import wrap
import numpy as np
# import demo_h5 as demo uncomment to display the data computed using demo_h5.py
import pandas as pd
from pann.pann_mel_inference import PannMelInference
from yamnet.yamnet_mel_inference import YamnetMelInference
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, multilabel_confusion_matrix, precision_score, recall_score, average_precision_score, precision_recall_curve, auc, plot_confusion_matrix
from classif_utils import accuracy_score_multilabel, precision_recall_curve_multilabel, sort_labels_by_score, load_sonyc_meta, load_urbansound8k_meta, accuracy_score_classification, folds_urbansound8k, folds_sonycust
import torch
import models as md
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import copy
import torch.nn.functional as F

class ClassifDataset(torch.utils.data.Dataset):
    def __init__(self, scores, gt, fnames=None):
        self.scores = scores
        self.gt = gt
        self.len_data = scores.shape[0]
        self.fnames = fnames
    def __getitem__(self, idx):

        x = torch.tensor(self.scores[idx])
        y = torch.tensor(self.gt[idx])

        if self.fnames is not None:
            fname = self.fnames[idx]
        else:
            fname = torch.tensor([])

        return (x, y, fname)

    def __len__(self):
        return self.len_data

class ClassifDatasetTrainer:
    def __init__(self, model_name, data_dir, audio_dir, models_path, scores_len, output_len, learning_rate=1e-3, dtype=torch.FloatTensor, 
                 ltype=torch.LongTensor):
        
        self.dtype = dtype
        self.ltype = ltype
        
        self.models_path = models_path
        
        if 'SONYC-UST' in audio_dir:
            self.loss_function = nn.BCELoss()
            self.dataset_fold = folds_sonycust(model_name, data_dir, audio_dir)
        if 'URBAN-SOUND-8K' in audio_dir:
            self.loss_function = nn.CrossEntropyLoss()
            self.dataset_fold = folds_urbansound8k(model_name, data_dir, audio_dir)


        self.scores_len = scores_len
        self.model = md.FC(scores_len, output_len)
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)
        self.model_fold = []
        self.train_fold = []
        self.eval_fold = []

    def train(self, batch_size=128, epoch=100, device=torch.device("cpu")):
        losses_train_fold = []
        losses_eval_fold = []

        for fold, (train_scores, eval_scores, train_groundtruth, eval_groundtruth, _ ) in enumerate(zip(self.dataset_fold[0], self.dataset_fold[1], self.dataset_fold[2], self.dataset_fold[3], self.dataset_fold[4])):  

            train_scores = np.array(train_scores)
            eval_scores = np.array(eval_scores)
            train_groundtruth = np.array(train_groundtruth)
            eval_groundtruth = np.array(eval_groundtruth)

            model = copy.deepcopy(self.model)
            optimizer = optim.Adam(params=model.parameters(), lr=self.learning_rate)
            loss_function = nn.BCELoss()
            model = model.to(device)
            
            cur_loss = 0

            train_dataset = ClassifDataset(train_scores, train_groundtruth)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False, drop_last=False)

            losses_train = []
            losses_eval = []

            for cur_epoch in range(epoch):
                tqdm_it=tqdm(train_dataloader, 
                            desc='TRAINING: Epoch {}, loss: {:.4f}'
                            .format(cur_epoch+1, cur_loss))
                for (x,y, _) in tqdm_it:
                    x = x.type(self.dtype)
                    y = y.type(self.dtype)
                    
                    x = x.to(device)
                    y = y.to(device)
                    
                    optimizer.zero_grad()

                    y_pred = model(x)

                    cur_loss = loss_function(y_pred,y)

                    cur_loss.backward()
                    
                    optimizer.step()
                    
                    cur_loss = float(cur_loss.data)

                    losses_train.append(cur_loss)
                    
                    tqdm_it.set_description('TRAINING: Epoch {}, loss: {:.4f}'
                                            .format(cur_epoch+1,cur_loss))                    

                loss_eval = self.validate(eval_scores, eval_groundtruth, model, batch_size=batch_size, device=device, label='EVALUATION')
                losses_eval.append(loss_eval)
            
            losses_train_fold.append(losses_train)
            losses_eval_fold.append(losses_eval)
            self.model_fold.append(model.state_dict())

        return(losses_train_fold, losses_eval_fold)

    def validate(self, eval_scores, eval_groundtruth, model, batch_size=64, device=torch.device("cpu"), label='VALIDATION', forced=False,
                    n_chunk=100):

        losses_valid = []

        valid_dataset = ClassifDataset(eval_scores, eval_groundtruth)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False, drop_last=False)
        tqdm_it=tqdm(valid_dataloader, desc=label+': Chunk {}/{}'.format(0,0))
        for (x,y, _) in tqdm_it:
            x = x.type(self.dtype)
            y = y.type(self.dtype)

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            
            cur_loss = self.loss_function(y_pred,y)

            losses_valid.append(cur_loss.detach())
            tqdm_it.set_description(label+': Chunk {}/{}'
                                            .format(0,0))  

        losses_valid = torch.Tensor(losses_valid)
        loss_valid = torch.mean(losses_valid)

        return loss_valid.detach().cpu().numpy()
    
    def evaluate(self, save_output_path, batch_size=1, device=torch.device("cpu")):

        loaded_model_fold = [copy.deepcopy(self.model) for k in range(len(self.model_fold))]
        for idx, state_dict in enumerate(self.model_fold):
            loaded_model_fold[idx].load_state_dict(state_dict)

        for fold, (_, eval_scores, _, eval_groundtruth, eval_fnames) in enumerate(zip(self.dataset_fold[0], self.dataset_fold[1], self.dataset_fold[2], self.dataset_fold[3], self.dataset_fold[4])):  
            
            model = loaded_model_fold[fold]

            eval_scores = np.array(eval_scores)
            eval_groundtruth = np.array(eval_groundtruth)
            eval_fnames = np.array(eval_fnames)
            
            eval_dataset = ClassifDataset(eval_scores, eval_groundtruth, eval_fnames)
            eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)

            tqdm_it=tqdm(eval_dataloader, 
                        desc='EVALUATING: ')
            for (x,y, fname) in tqdm_it:
                x = x.type(self.dtype)
                y = y.type(self.dtype)
                
                x = x.to(device)
                y = y.to(device)
                
                y_pred = model(x)

                np.save(save_output_path+'___'+fname[0]+'.npy', y_pred.detach().numpy())
                                    
        return()

def get_model_parameters(model_name, data_dir, audio_dir):
    trainer = ClassifDatasetTrainer()
    train_folds, eval_folds = folds_urbansound8k(model_name, data_dir, audio_dir)


def get_score(model_name, data_dir, audio_dir, deep_name=None, top=8):
    """
    Calculate evaluation metrics for a given model. The same calculation is made
    for multi-class (UrbanSound8k) and for multi-lab (SONYC-UST) datasets. For
    UrbanSound8k, we condider that a muli-class problem is a specific case of
    multilabel with only one category set to 1. This allows the calculation
    of multilabel metrics on UrbanSound8k. To calculate classification metrics
    on UrbanSound8k, we get back to a classification problem by setting the
    top predicted logit to 1.

    Args:
        model_name (str): Name of the model.
        data_dir (str): Directory containing the data.
        audio_dir (str): Directory containing the audio files.
        deep_name (str, optional): Deep name. Defaults to None.
        top (int, optional): Number of top labels to consider. Defaults to 8.

    Returns:
        tuple: Tuple containing accuracy of classification and macro average precision-recall curve (AUPRC).

    """
    if 'SONYC-UST-LOGITS' in data_dir:
        df, df_subclasses, weights, _ = load_sonyc_meta(audio_dir, data_dir, subset=['test'], verbose=True)
        idx_start = 1
        top = 8
        multilabel=True
    if 'URBAN-SOUND-8K-LOGITS' in data_dir:
        df, df_subclasses, weights, _ = load_urbansound8k_meta(audio_dir, data_dir, verbose=True)
        idx_start = 0
        top = 1
        multilabel=False
    print('XXXXXXXXXXXX')
    print(data_dir)
    key_list = list(df.columns)
    idx_dict = {key: [] for key in df.columns}

    if deep_name is None:

        if 'pann' in model_name:
            evaluator = PannMelInference(verbose=False)
        if 'yamnet' in model_name:
            evaluator = YamnetMelInference(verbose=False)

        if top == -1:
            thrsh = evaluator.threshold
        labels = evaluator.labels_str
        df_subclasses['labels_index'] = df_subclasses['class'].apply(lambda x: labels.index(x) if x in labels else -1)
        #df_subclasses.dropna(inplace=True)
        df_subclasses = df_subclasses[df_subclasses['labels_index']!=-1]
        df_subclasses['sub_class_str'] = df_subclasses['sub_class'].apply(lambda x: df.columns[x-idx_start] if x!=-1 else 'None')

        for k, key in enumerate(idx_dict):
            idx_dict[key] = df_subclasses[df_subclasses['sub_class'] == k+1]['labels_index'].to_list()
        
        cols = df.columns.to_numpy()

        for f_audio in df.index:
            f_name = data_dir + "/" + f_audio[:-4] + "_" + model_name + ".npy"
            logit = np.load(f_name)
            df_logit = df_subclasses.copy()
            df_logit['logit'] = logit
            df_classes_logit = df_logit[df_logit['sub_class_str'] != 'None'].groupby('sub_class_str').apply(
                lambda x: pd.Series({
                    'class': x['sub_class_str'].iloc[0],
                    'sub_class_str': x['sub_class_str'].iloc[0],
                    'labels_index': x['labels_index'].iloc[0],
                    'logit': x['logit'].max()
                })
                )
            grouped = pd.concat([df_classes_logit, df_logit[df_logit['sub_class_str'] == 'None']])
            if top != -1:
                if multilabel == True:
                    _, sorted_labels, _ = sort_labels_by_score(grouped['logit'].to_numpy(), grouped['class'].to_numpy())
                    top_sorted_labels = sorted_labels[:top]
                else:
                    _, sorted_labels, _ = sort_labels_by_score(df_classes_logit['logit'].to_numpy(), df_classes_logit['class'].to_numpy())
                    top_sorted_labels = sorted_labels[:1]

            for col in cols:
                df.loc[f_audio, 'pred_'+col] = grouped.loc[col, "logit"]
            for col in cols:
                df.loc[f_audio, 'pred_binary_'+col] = int(col in top_sorted_labels)

    else:
        cols = df.columns
        for f_audio in df.index:
            f_name = deep_name + '___' + f_audio[:-4] + ".npy"
            output = np.load(f_name)[0]
            if multilabel:
                output_binary = np.where(output>=0.5, 1, 0)
            else:
                output_binary = np.where(output == np.max(output), 1, 0)
            #get one column for the prediction of each class. The raw
            #prediction (ex: [0.8, 0.1, 0.7, ..., 0.5] is put in "pred_xxx" column, while the binary
            #prediction (ex: [1, 0, 1, ..., 1] is put in "pred_binary_xxx")
            for k, col in enumerate(cols):
                df.loc[f_audio, 'pred_'+col] = output[k]
            for k, col in enumerate(cols):
                df.loc[f_audio, 'pred_binary_'+col] = output_binary[k]


    # get the groundtruth outputs, the predicted ones, and the binary predicted ones
    # This is used for multilabel metrics only.
    y_gt = df.iloc[:, 0:len(cols)].to_numpy()
    y_pred = df.iloc[:, len(cols):2*len(cols)].to_numpy()
    y_pred_binary = df.iloc[:, 2*len(cols):3*len(cols)].to_numpy()

    # get the groundtruth and the prediction for classification.
    # This is used for classification metrics only
    df_gt = df.iloc[:, 0:len(cols)]
    df_gt['gt_classif'] = df_gt.apply(lambda x: x.idxmax(), axis=1)
    df_gt['gt_classif'] = df_gt['gt_classif'].apply(lambda x: df_gt.columns.get_loc(x))

    df_pred = df.iloc[:, 2*len(cols):3*len(cols)]
    df_pred['pred_classif'] = df_pred.apply(lambda x: x.idxmax(), axis=1)
    df_pred['pred_classif'] = df_pred['pred_classif'].apply(lambda x: df_pred.columns.get_loc(x))

    y_gt_classif = df_gt['gt_classif'].to_numpy()
    y_pred_classif = df_pred['pred_classif'].to_numpy()

    # print('F1 score')
    # f1 = f1_score(y_gt, y_pred_binary, average=None)
    # print(f1)
    # print('Accuracy')
    # acc = accuracy_score_multilabel(y_gt, y_pred_binary)
    # print(acc)
    # print(np.mean(acc))
    # print('Overall Accuracy')
    # acc = accuracy_score_classification(y_gt, y_pred_binary, weights)
    # print(acc)
    print('Accuracy for multi-class classification (useless for SONYC-UST)')
    acc_classif = accuracy_score(y_gt_classif, y_pred_classif)
    print(acc_classif)
    # print('CF matrix classif')
    # cf_matrix = confusion_matrix(y_gt_classif, y_pred_classif)
    # print(cf_matrix)

    # print('Precision')
    # prec = precision_score(y_gt, y_pred_binary, average=None)
    # print(prec)
    # print(np.mean(prec))
    # print('Recall')
    # rec = recall_score(y_gt, y_pred_binary, average=None)
    # print(rec)
    # print(np.mean(rec))
    # print('Average Precision Score Macro')
    # avgp = average_precision_score(y_gt, y_pred_binary, average=None)
    # avgp_micro = average_precision_score(y_gt, y_pred_binary, average='micro')
    # print(avgp)
    # print(np.mean(avgp))
    # print(avgp_micro)

    print('macro AUPRC for multi-label classification (useless for UrbanSound8k)')
    precision, recall, thresholds, auc_precision_recall = precision_recall_curve_multilabel(y_gt, y_pred)
    print(auc_precision_recall)
    macro_auprc = np.mean(auc_precision_recall)
    print(macro_auprc)

    return(acc_classif, macro_auprc)
