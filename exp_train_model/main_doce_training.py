import numpy as np
import doce
from pathlib import Path
import time
import torch
import os
import sys
# Add the parent directory of the project directory to the module search path
project_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_parent_dir)
from doce.setting import Setting
import utils.util as ut
import experiment_manager

torch.manual_seed(0)

if torch.cuda.is_available():
    # Set the random seed for GPU (if available)
    torch.cuda.manual_seed(0)

# define the experiment
experiment = doce.Experiment(
  name = "train_exp",
  purpose = 'experiment for spectral transcoder',
  author = 'Modan Tailleur',
  address = 'modan.tailleur@ls2n.fr',
)

#########################
#########################
# EXPERIMENT PATH

# set acces paths (here only storage is needed)
#general
exp_path = './spectral_transcoder_data/training_experiments/'

#########################
######################################
# PROJECT TRAIN DATA PATH

#general
PROJECT_DATA_PATH = Path('./spectral_transcoder_data/')

if not os.path.exists(exp_path):
    # Create the directory recursively
    os.makedirs(exp_path)

experiment.set_path('output', exp_path+experiment.name+'/', force=True)
experiment.set_path('duration', exp_path+experiment.name+'/duration/', force=True)
experiment.set_path('model', exp_path+experiment.name+'/model/', force=True)
experiment.set_path('loss', exp_path+experiment.name+'/loss/', force=True)
experiment.set_path('evaluation', exp_path+experiment.name+'/evaluation/', force=True)
experiment.set_path('metric', exp_path+experiment.name+'/metric/', force=True)

experiment.add_plan('reference',
  ts = 0,
  step = ['evaluate', 'metric'],
  transcoder = ['oracle', 'pinv'],
  dataset = ['TEST', 'full', 'urban', 'urban_mel_time', 'urban_mel_freq'],
  classifier = ['YamNet','PANN'],
)

experiment.add_plan('mlp',
  ts = 0,
  step = ['train', 'evaluate', 'metric'],
  transcoder = ['mlp', 'mlp_pinv'],
  dataset = ['TEST', 'full', 'urban'],
  classifier = ['YamNet','PANN'],
  epoch = [1, 10, 200],
  learning_rate = [-3, -4, -5],
  hidden1 = [100, 300],
  hidden2 = [1000, 3000],
)

experiment.add_plan('cnn',
  ts = 0,
  step = ['train', 'evaluate', 'metric'],
  transcoder = ['cnn_pinv'],
  dataset = ['TEST', 'full', 'urban', 'urban_mel_time', 'urban_mel_freq'],
  classifier = ['YamNet','PANN'],
  epoch = [1, 10, 100, 200],
  learning_rate = [-3, -4, -5],
  kernel_size = [3, 5],
  nb_layers = [1, 3, 5],
  dilation = [0, 1, 2, 3],
  nb_channels = [16, 64]
)

experiment.add_plan('hybridts',
  ts = 1,
  step = ['train', 'evaluate', 'metric'],
  transcoder = ['cnn_pinv'],
  dataset = ['TEST', 'full', 'urban'],
  classifier = ['YamNet','PANN'],
  epoch = [1, 5, 10, 20, 50, 100, 200, 400],
  learning_rate = [-3, -4, -5],
  kernel_size = [5],
  nb_layers = [5],
  dilation = [0, 1],
  nb_channels = [64],
  prop_logit = [100, 50]
)

experiment.add_plan('ts',
  ts = 1,
  step = ['train', 'evaluate', 'metric'],
  #self means retrain YamNet if classifier is YamNet, retrain PANN if classifier is PANN
  transcoder = ["self", "effnet_b0", "effnet_b7"],
  dataset = ['TEST', 'full', 'urban'],
  classifier = ['YamNet','PANN'],
  learning_rate = [-3, -4, -5],
  epoch = [1, 5, 10, 20, 50, 100, 150, 200, 300, 400]
)

###############################
# metrics from loss functions
################################

experiment.set_metric(
  name = 'avg_mels_mse',
  path = 'metric',
  output = 'mel_mse',
  func = np.mean,
  significance = True,
  percent=False,
  lower_the_better=True,
  precision=2
  )

experiment.set_metric(
  name = 'avg_logit_bce',
  path = 'metric',
  output = 'logit_bce',
  func = np.mean,
  significance = True,
  percent=False,
  precision=6,
  lower_the_better=True
  )

###############################
# other metrics
################################

experiment.set_metric(
  name = 'avg_logit_kl',
  path = 'metric',
  output = 'logit_kl',
  func = np.mean,
  significance = True,
  percent=False,
  precision=6,
  lower_the_better=True
  )

experiment.set_metric(
  name = 'ptopafc',
  path = 'metric',
  output = 'ptopafc',
  func = np.mean,
  percent=True,
  precision=3,
  higher_the_better=True,
  significance=True
  )

class SettingExp(Setting):
    def __init__(self, setting, experiment, project_data_path, force_cpu=False):
        """
        Represents an experimental setting for the transcoder project. Some attributes are
        directly from doce settings, others are created in the initialisation, and depend
        on the values of the doce settings.
        
        NB: Doce setting attributes have the particularity of being "objects" types, this lead to a lot of bugs,
        especially when trying to store their values into yaml files. In this class, correct
        data types are reattributed to every doce setting attribute. 

        Args:
            setting: A doce setting object.
            experiment: The doce experiment object.
            project_data_path: Path to the project's data directory. This path must contain the data created with create_mel_tho_dataset.
            force_cpu: Whether to force CPU usage instead of GPU (default: False).
        """

        # parameter to switch to test with non dB input. Experiments showed that input_is_dB should always be set to true
        self.input_is_db = True


        self.batch_size = 64
        self.plan_name = experiment.get_current_plan().get_name()
        self.project_data_path = project_data_path
        self.ts = getattr(setting, 'ts', None)
        self.step = getattr(setting, 'step', None)
        self.transcoder = str(getattr(setting, 'transcoder', None)) if getattr(setting, 'transcoder', None) is not None else None
        self.dataset = getattr(setting, 'dataset', None)
        self.classifier = str(getattr(setting, 'classifier', None)) if getattr(setting, 'classifier', None) is not None else None

        #deep
        self.epoch = int(getattr(setting, 'epoch', None)) if getattr(setting, 'epoch', None) is not None else None
        self.learning_rate = 10**float(getattr(setting, 'learning_rate', None)) if getattr(setting, 'learning_rate', None) is not None else None            

        #mlp
        self.hidden1 = int(getattr(setting, 'hidden1', None)) if getattr(setting, 'hidden1', None) is not None else None
        self.hidden2 = int(getattr(setting, 'hidden2', None)) if getattr(setting, 'hidden2', None) is not None else None

        #cnn
        self.kernel_size = int(getattr(setting, 'kernel_size', None)) if getattr(setting, 'kernel_size', None) is not None else None
        self.nb_layers = int(getattr(setting, 'nb_layers', None)) if getattr(setting, 'nb_layers', None) is not None else None
        self.dilation = int(getattr(setting, 'dilation', None)) if getattr(setting, 'dilation', None) is not None else None
        self.nb_channels = int(getattr(setting, 'nb_channels', None)) if getattr(setting, 'nb_channels', None) is not None else None

        #hybridts
        self.prop_logit = int(getattr(setting, 'prop_logit', None)) if getattr(setting, 'prop_logit', None) is not None else None

        #manage case where no mels are utilise for the reconstruction
        if self.transcoder in ["effnet_b0", "effnet_b7", "self"]:
            self.no_mels = True
        else:
            self.no_mels = False

        #manage gpu
        useCuda = torch.cuda.is_available() and not force_cpu
        if useCuda:
            print('Using CUDA.')
            self.dtype = torch.cuda.FloatTensor
            self.ltype = torch.cuda.LongTensor
            #MT: add
            self.device = torch.device("cuda:0")
        else:
            print('No CUDA available.')
            self.dtype = torch.FloatTensor
            self.ltype = torch.LongTensor
            #MT: add
            self.device = torch.device("cpu")

        #paths
        self.model_path = experiment.path.model

        model_mel_path = experiment.path.evaluation+setting.replace('step', 'evaluate').identifier()+'_mel'
        model_logit_path = experiment.path.evaluation+setting.replace('step', 'evaluate').identifier()+'_logit'
        model_logit_tvb_path = experiment.path.evaluation+setting.replace('step', 'evaluate').identifier()+'_logit_tvb'

        self.outputs_path =	{
            "mels": model_mel_path ,
            "logits": model_logit_path,
            "logits_tvb": model_logit_tvb_path
            }
        
        str_oracle = experiment.path.evaluation+doce.Setting(experiment.reference, [0, 'evaluate', 'oracle', setting.dataset, setting.classifier], positional=False).identifier()
        oracle_mel_path = str_oracle +'_mel'
        oracle_logit_path = str_oracle + '_logit'
        oracle_logit_tvb_path = str_oracle + '_logit_tvb'

        self.outputs_oracle_path =	{
            "mels": oracle_mel_path ,
            "logits": oracle_logit_path,
            "logits_tvb": oracle_logit_tvb_path
            }

        #manage paths depending on factors
        if self.step == "metric":
            model_mel_path = experiment.path.evaluation+setting.replace('step', 'evaluate').identifier()+'_mel'
            model_logit_path = experiment.path.evaluation+setting.replace('step', 'evaluate').identifier()+'_logit'
            model_logit_tvb_path = experiment.path.evaluation+setting.replace('step', 'evaluate').identifier()+'_logit_tvb'
            
            outputs_path =	{
                "mels": model_mel_path ,
                "logits": model_logit_path,
                "logits_tvb": model_logit_tvb_path
                }
        
        #model name, and losses path definition
        if self.step != "train":
            self.losses_path = None
            if self.transcoder in ['mlp', 'mlp_pinv', 'cnn_pinv', 'self', 'effnet_b0', 'effnet_b7']:
                self.model_name = setting.replace('step', 'train').identifier()+'_model'
            else:
                self.model_name = None
        else:
            self.losses_path = self.project_data_path / 'losses' / ('losses_' + self.transcoder)
            self.model_name = setting.identifier()+'_model'

        #The following parameters (tho_freq, tho_time, mel_template) are just a test to check what is the most difficult to 
        # retrieve by the generative model: mel frequencies or temporal windows
        #if tho_freq, means that third-octave frequencies are kept as third-octave frequencies. If not we use the one of Mel representation
        # if tho_time, means that third-octave time windows are kepts as third-octave time windows. If not, we use the one of Mel representation.
        self.tho_freq = True
        self.tho_time = True
        self.mel_template = None
        if setting.dataset == 'full':
            self.config = "Dcase-Task1-full"
        if setting.dataset == 'urban':
            self.config = "Dcase-Task1-urban"    
        if setting.dataset == 'TEST':
            self.config = "Dcase-Task1-TEST"    
        if setting.dataset == 'urban_mel_time':
            self.tho_time = False
            if setting.classifier == 'PANN':
                self.config = "Dcase-Task1-Pann-UrbanDataset-MEL-TIME-PANN"
                self.mel_template = 'PANN'

        if setting.dataset == 'urban_mel_freq':
            self.tho_freq = False
            if setting.classifier == 'PANN':
                self.config = "Dcase-Task1-Pann-UrbanDataset-MEL-FREQ-PANN"
                self.mel_template = 'PANN'

        #data settings
        self.losses_path = self.project_data_path / 'losses' / ('losses_' + self.transcoder)
        self.setting_data_path = self.project_data_path / 'data_settings' 
        self.setting_data = ut.load_settings(Path(self.setting_data_path / (self.config+'_settings'+'.yaml')))

def step(setting, experiment):
    
    print('XXXXXXXX ONGOING SETTING XXXXXXXX')
    print(setting.identifier())
    start_time = time.time()

    setting_exp = SettingExp(setting, experiment, PROJECT_DATA_PATH)

    if setting_exp.step == "train":
        if setting_exp.transcoder in ['mlp', 'mlp_pinv', 'cnn_pinv', 'self', 'effnet_b0', 'effnet_b7']:
            losses, train_duration = experiment_manager.train_dl_model(setting_exp)
            
            #saving losses in the loss folder
            for key, arr in losses.items():
                np.save(experiment.path.loss+setting.identifier()+'_'+key+'.npy', arr)

    if setting_exp.step == "evaluate":
        
        if setting_exp.transcoder in ['mlp', 'mlp_pinv', 'cnn_pinv', 'self', 'effnet_b0', 'effnet_b7']:
            experiment_manager.evaluate_dl_model(setting_exp)

        if setting_exp.transcoder == 'pinv':
            experiment_manager.evaluate_pinv(setting_exp)
        
        if setting_exp.transcoder == 'oracle':
            experiment_manager.evaluate_oracle(setting_exp)
    
    if setting_exp.step == "metric":
          
        metrics, others = experiment_manager.compute_metrics(setting_exp)
        
        for key, arr in metrics.items():
          np.save(experiment.path.metric+setting.identifier()+'_'+key+'.npy', arr)
        
        for key, arr in others.items():
          np.save(experiment.path.metric+setting.identifier()+'_'+key+'.npy', arr)

    duration = time.time() - start_time

    if setting_exp.step == "train":
      np.save(experiment.path.duration+setting.identifier()+'_train_duration.npy', train_duration)
      print("--- %s seconds ---" % (train_duration))

    np.save(experiment.path.duration+setting.identifier()+'_duration.npy', duration)
    print("--- %s seconds ---" % (duration))
        
# invoke the command line management of the doce package
if __name__ == "__main__":
  doce.cli.main(experiment = experiment, func=step)
