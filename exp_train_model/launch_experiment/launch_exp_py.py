import argparse
import os
import sys
# Add the parent directory of the project directory to the module search path
project_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_parent_dir)

def main(config):

  launch_dict = {}

  if config.exp_type == 'restricted':

    launch_dict['oracle'] = 'python3 exp_train_model/main_doce_training.py -s reference/transcoder=oracle+dataset=' + config.dataset + ' -c'

    launch_dict['pinv'] = 'python3 exp_train_model/main_doce_training.py -s reference/transcoder=pinv+dataset=' + config.dataset + ' -c'

    launch_dict['cnn'] = 'python3 exp_train_model/main_doce_training.py -s cnn/transcoder=cnn_pinv+dataset=' + config.dataset + '+epoch=' + config.epoch + '+kernel_size=5+nb_layers=5+dilation=0+nb_channels=64+learning_rate=-3 -c'

    launch_dict['hybridts'] = 'python3 exp_train_model/main_doce_training.py -s hybridts/transcoder=cnn_pinv+dataset=' + config.dataset + '+epoch=' + config.epoch + '+kernel_size=5+nb_layers=5+dilation=0+nb_channels=64+prop_logit=100+learning_rate=-3 -c'

    launch_dict['self'] = 'python3 exp_train_model/main_doce_training.py -s ts/dataset=' + config.dataset + '+epoch=' + config.epoch + '+transcoder=self+learning_rate=-4' + ' -c'

    launch_dict['effnet_b0'] = 'python3 exp_train_model/main_doce_training.py -s ts/dataset=' + config.dataset + '+epoch=' + config.epoch + '+transcoder=effnet_b0+learning_rate=-5' + ' -c'

    launch_dict['effnet_b7'] = 'python3 exp_train_model/main_doce_training.py -s ts/dataset=' + config.dataset + '+epoch=' + config.epoch + '+transcoder=effnet_b7+learning_rate=-5' + ' -c'

  if config.exp_type == 'detailed':

    launch_dict['oracle'] = 'python3 exp_train_model/main_doce_training.py -s reference/transcoder=oracle+dataset=' + config.dataset + ' -c'

    launch_dict['pinv'] = 'python3 exp_train_model/main_doce_training.py -s reference/transcoder=pinv+dataset=' + config.dataset + ' -c'

    launch_dict['mlp'] = 'python3 exp_train_model/main_doce_training.py -s mlp/transcoder=mlp_pinv+dataset=' + config.dataset + '+epoch=' + config.epoch + ' -c'

    launch_dict['cnn'] = 'python3 exp_train_model/main_doce_training.py -s cnn/transcoder=cnn_pinv+dataset=' + config.dataset + '+epoch=' + config.epoch + ' -c'

    launch_dict['hybridts'] = 'python3 exp_train_model/main_doce_training.py -s hybridts/transcoder=cnn_pinv+dataset=' + config.dataset + '+epoch=' + config.epoch + '+prop_logit=100 -c'

    launch_dict['ts'] = 'python3 exp_train_model/main_doce_training.py -s ts/dataset=' + config.dataset + '+epoch=' + config.epoch + ' -c'

  for key, value in launch_dict.items():
    print('EXPERIMENT: ' + str(key))
    os.system(value)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='full', help='Dataset used: full, urban , test')
    parser.add_argument('--epoch', type=int, default=200, help='Number of epochs for training: 1, 10, 50, 100, 200')
    parser.add_argument('--exp_type', type=str, default='restricted', help='Experiment type: detailed, restricted. Restricted only launches the experiments that are shown on the Spectral Transcoder Dcase2023 paper')
    config = parser.parse_args()
    main(config)
