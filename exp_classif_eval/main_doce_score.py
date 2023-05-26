import numpy as np
import doce
import torch
import os
import sys
# Add the parent directory of the project directory to the module search path
project_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_parent_dir)
import pickle 
import score_metric_manager
from score_metric_manager import ClassifDatasetTrainer

torch.manual_seed(0)
# import sys
# # caution: path[0] is reserved for script path (or '' in REPL)
# sys.path.append('./yamnet/')

# define the experiment
experiment = doce.Experiment(
  name = 'classif_exp',
  purpose = 'calculates the correlation in time of presence of tvb for the outputs of pann, yamnet and for the tfsd',
  author = 'Modan Tailleur',
  address = 'modan.tailleur@ls2n.fr',
)

########## ACCESS PATH ##################

#general
exp_path = './spectral_transcoder_data/classification_experiments/'
if not os.path.exists(exp_path):
    # Create the directory recursively
    os.makedirs(exp_path)
    
#########################################


experiment.set_path('output',  exp_path+experiment.name+'/', force=True)
experiment.set_path('model', exp_path+experiment.name+'/model/', force=True)
experiment.set_path('loss', exp_path+experiment.name+'/loss/', force=True)
experiment.set_path('predictions', exp_path+experiment.name+'/predictions/', force=True)
experiment.set_path('metrics', exp_path+experiment.name+'/metrics/', force=True)

experiment.add_plan('reference',
  deep=0,
  step = ['metric'],
  transcoder = ['self', 'cnn'],
  classifier = ['PANN', 'YamNet'],
  dataset = ['SONYC-UST', 'URBAN-SOUND-8K'],
)

experiment.add_plan('deep',
  deep=1,
  step = ['train', 'eval', 'metric'],
  transcoder = ['self', 'cnn'],
  classifier = ['PANN', 'YamNet'],
  dataset = ['SONYC-UST', 'URBAN-SOUND-8K'],
)

#traffic correlation
experiment.set_metric(
  name = 'acc_classif',
  path = 'metrics',
  significance = True,
  percent=True,
  higher_the_better=True,
  precision=3
  )

experiment.set_metric(
  name = 'macro_auprc',
  path = 'metrics',
  significance = True,
  percent=False,
  higher_the_better=True,
  precision=2
  )

def step(setting, experiment):
    
    print('XXXXXXXX ONGOING EXPERIMENT XXXXXXXX')
    print(setting.identifier())
    plan_name = experiment.get_current_plan().get_name()

    # choose the correct audio directory data_dir 
    if setting.dataset == 'SONYC-UST':
        data_dir = os.path.join(os.path.dirname(os.path.dirname(exp_path)), 'data', 'SONYC-UST-LOGITS')
        audio_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(exp_path))), 'SONYC-UST')
        output_len = 8
        epoch = 30
        batch_size = 32
    if setting.dataset == 'URBAN-SOUND-8K':
        data_dir = os.path.join(os.path.dirname(os.path.dirname(exp_path)), 'data', 'URBAN-SOUND-8K-LOGITS')
        audio_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(exp_path))), 'URBAN-SOUND-8K')
        output_len = 10
        epoch = 30
        batch_size = 32
    if setting.classifier == 'YamNet':
        scores_len=521
        if setting.transcoder == 'self':
            model_name = 'yamnet_gt'
        if setting.transcoder == 'cnn':
            model_name = 'yamnet_cnn'
    if setting.classifier == 'PANN':
        scores_len=527
        if setting.transcoder == 'self':
            model_name = 'pann_gt'
        if setting.transcoder == 'cnn':
            model_name = 'pann_cnn'

    if setting.step == 'metric':
        deep_name = None
        if plan_name == 'deep':
            str_deep = doce.Setting(experiment.deep, [setting.deep, 'eval', setting.transcoder, setting.classifier, setting.dataset], positional=False).identifier()
            deep_name = experiment.path.predictions + str_deep
        acc_classif, macro_auprc = score_metric_manager.get_score(model_name, data_dir, audio_dir, deep_name)
        np.save(experiment.path.metrics+setting.identifier()+'_acc_classif.npy', acc_classif)
        np.save(experiment.path.metrics+setting.identifier()+'_macro_auprc.npy', macro_auprc)

    if setting.step == 'train':
        if plan_name == 'deep':
            trainer = ClassifDatasetTrainer(model_name, data_dir, audio_dir, experiment.path.model, scores_len, output_len)
            losses_train_fold, losses_eval_fold = trainer.train(epoch=epoch, batch_size=batch_size)

            np.save(experiment.path.loss+setting.identifier()+'_loss_train.npy', losses_train_fold)
            np.save(experiment.path.loss+setting.identifier()+'_loss_eval.npy', losses_eval_fold)

            with open((experiment.path.model+setting.identifier()+'_model_fold'), 'wb') as f:
                pickle.dump(trainer.model_fold, f)

    if setting.step == 'eval':
        if plan_name == 'deep':
            str_model = doce.Setting(experiment.deep, [setting.deep, 'train', setting.transcoder, setting.classifier, setting.dataset], positional=False).identifier()
            model_fold_name = str_model+'_model_fold'
            with open(experiment.path.model+model_fold_name, 'rb') as f:
                    model_fold = pickle.load(f)

            trainer = ClassifDatasetTrainer(model_name, data_dir, audio_dir, experiment.path.model, scores_len, output_len)
            trainer.model_fold = model_fold
            trainer.evaluate(experiment.path.predictions+setting.identifier())
        
# invoke the command line management of the doce package
if __name__ == "__main__":
  doce.cli.main(experiment = experiment, func=step)