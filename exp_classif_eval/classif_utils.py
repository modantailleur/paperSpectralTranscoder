import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, multilabel_confusion_matrix, precision_score, recall_score, precision_recall_curve, auc, PrecisionRecallDisplay

def accuracy_score_multilabel(y_gt, y_pred, average=None):
    X = []
    cf_matrix = multilabel_confusion_matrix(y_gt, y_pred)
    cf_norm = cf_matrix.astype('float') / cf_matrix.sum(axis=2)[:, np.newaxis]
    for k in range(y_gt.shape[1]):
        X.append(np.mean([cf_norm[k, 0, 0], cf_norm[k, 1, 1]]))
    if average == 'macro':
        out = X.mean()
    else:
        out = X
    return(out)

def accuracy_score_classification(y_gt, y_pred, weights):
    X = []
    total_accuracy = 0
    if weights is not None:
        for k in range(y_gt.shape[1]):
            total_accuracy += accuracy_score(y_gt[:, k], y_pred[:, k])*weights[k]
    return(total_accuracy)

def precision_recall_curve_multilabel(y_gt, y_pred, average=None):
    precision_l = []
    recall_l = []
    thresholds_l = []
    auc_l = []
    for k in range(y_gt.shape[1]):
        precision, recall, thresholds = precision_recall_curve(y_gt[:, k], y_pred[:, k])
        auc_l.append(auc(recall, precision))
        
        precision_l.append(precision)
        recall_l.append(recall)
        thresholds_l.append(thresholds)

    return(precision_l, recall_l, thresholds_l, auc_l)

def load_sonyc_meta(audio_dir, data_dir=None, subset=None, verbose=False):
    #subset is a list of subsets (ex: ['train', 'valid'])
    #load sonyc metadata
    df = pd.read_csv(audio_dir + "/annotations.csv")

    if subset:
        df = df[df['split'].isin(subset)]

    df_0 = df[df['split'] == 'test']
    df_rest =  df[df['split'] != 'test']
    df_0 = df_0[df_0['annotator_id'] == 0]
    df = pd.concat([df_0, df_rest])
    weights = None
    folds = df.groupby('audio_filename').apply(lambda x: x['split'].iloc[0])
    df = df.groupby('audio_filename', as_index=False).apply(lambda x: x.iloc[:, -8:].replace(-1, np.nan).mean(skipna=True))
    df.set_index('audio_filename', inplace=True)
    df = df.applymap(lambda x: 0 if x < 0.5 else 1)
    if data_dir:
        #load excel file with equivalences for Pann and Yamnet
        df_subclasses = pd.read_excel("exp_classif_eval/sub_classes_sonyc_ust.xlsx", index_col=0, usecols=[0,1])
        df_subclasses['sub_class'].fillna(-1, inplace=True)
        df_subclasses['sub_class'] = df_subclasses['sub_class'].astype(int)
        df_subclasses['class'] = df_subclasses.index
    else:
        df_subclasses = None

    if verbose:
        df_subclasses_raw = df_subclasses.reset_index(drop=True)
        df_subclasses_raw['class'] = df_subclasses_raw.index
        df_indexes = df_subclasses_raw.groupby('sub_class').agg(list)
        df_indexes.columns = ['indexes']
        df_indexes = df_indexes.reset_index()

        print('INDEXES OF SUB CLASS')
        print(df_indexes)
        for k in range(len(df_indexes)):
            print(df_indexes.iloc[k]['sub_class'])
            print(df_indexes.iloc[k]['indexes'])


    return(df, df_subclasses, weights, folds)

def load_urbansound8k_meta(audio_dir, data_dir=None, subset=None, verbose=True):
    df = pd.read_csv(audio_dir + "/metadata/UrbanSound8K_recalculated.csv")
    df = df[df['duration'] >= 1]
    df['full_class'] = df.apply(lambda x: f"{x['classID']}_{x['class']}", axis=1)
    weights = (df["full_class"].value_counts().sort_index()/len(df)).to_numpy()
    df = pd.get_dummies(df, columns=['full_class'], prefix='', prefix_sep='')
    df.set_index('slice_file_name', inplace=True)
    folds = df['fold']
    df = df.iloc[:, -10:]

    if data_dir:
        #load excel file with equivalences for Pann and Yamnet
        df_subclasses = pd.read_excel("exp_classif_eval/sub_classes_urbansound8k.xlsx", index_col=0, usecols=[0,1])
        df_subclasses['sub_class'].fillna(-1, inplace=True)
        df_subclasses['sub_class'] = df_subclasses['sub_class'].astype(int)
        df_subclasses['class'] = df_subclasses.index
    else:
        df_subclasses = None

    if verbose:
        
        df_subclasses_raw = df_subclasses.reset_index(drop=True)
        df_subclasses_raw['class'] = df_subclasses_raw.index 
        df_indexes = df_subclasses_raw.groupby('sub_class').agg(list)
        df_indexes.columns = ['indexes']
        df_indexes = df_indexes.reset_index()

        print('INDEXES OF SUB CLASS')
        print(df_indexes)
        for k in range(len(df_indexes)):
            print(df_indexes.iloc[k]['sub_class'])
            print(df_indexes.iloc[k]['indexes'])


    return(df, df_subclasses, weights, folds)

def folds_sonycust(model_name, data_dir, audio_dir):
    df, _, _, folds = load_sonyc_meta(audio_dir, data_dir)
    scores_train_folds = [ [] for k in range(1)]
    scores_eval_folds = [ [] for k in range(1)]
    gt_train_folds = [ [] for k in range(1)]
    gt_eval_folds = [ [] for k in range(1)]
    eval_fnames = [ [] for k in range(1)]

    for f_audio in df.index:
        f_name = data_dir + "/" + f_audio[:-4] + "_" + model_name + ".npy"
        logit = np.load(f_name)
        fold = folds[f_audio]
        gt = df.loc[f_audio].to_numpy()
        if fold == 'train':
            scores_train_folds[0].append(logit)
            gt_train_folds[0].append(gt)
        if fold == 'test':
            scores_eval_folds[0].append(logit)
            gt_eval_folds[0].append(gt)
            eval_fnames[0].append(f_audio[:-4])

    return(scores_train_folds, scores_eval_folds, gt_train_folds, gt_eval_folds, eval_fnames)


def folds_urbansound8k(model_name, data_dir, audio_dir):
    df, _, _, folds = load_urbansound8k_meta(audio_dir, data_dir)
    n_folds = 10
    scores_train_folds = [ [] for k in range(n_folds)]
    scores_eval_folds = [ [] for k in range(n_folds)]
    gt_train_folds = [ [] for k in range(n_folds)]
    gt_eval_folds = [ [] for k in range(n_folds)]
    eval_fnames = [ [] for k in range(n_folds)]

    for f_audio in df.index:
        f_name = data_dir + "/" + f_audio[:-4] + "_" + model_name + ".npy"
        logit = np.load(f_name)
        fold = folds[f_audio]-1
        gt = df.loc[f_audio].to_numpy()
        for k in range(n_folds):
            if k != fold:
                scores_train_folds[k].append(logit)
                gt_train_folds[k].append(gt)
            else:
                scores_eval_folds[k].append(logit)
                gt_eval_folds[k].append(gt)
                eval_fnames[k].append(f_audio[:-4])

    return(scores_train_folds, scores_eval_folds, gt_train_folds, gt_eval_folds, eval_fnames)