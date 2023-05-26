import argparse
import os
import sys
# Add the parent directory of the project directory to the module search path
project_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_parent_dir)

def main(config):

  oracle = 'python3 exp_train_model/main_doce_training.py -s reference/transcoder=oracle+dataset=TEST -c'

  pinv = 'python3 exp_train_model/main_doce_training.py -s reference/transcoder=pinv+dataset=TEST -c'

  mlp = 'python3 exp_train_model/main_doce_training.py -s mlp/transcoder=mlp_pinv+dataset=TEST+classifier=YamNet+epoch=1+learning_rate=-3+hidden1=100+hidden2=1000 -c'

  cnn = 'python3 exp_train_model/main_doce_training.py -s cnn/transcoder=cnn_pinv+dataset=TEST+classifier=YamNet+epoch=1+learning_rate=-3+kernel_size=3+nb_layers=1+dilation=0+nb_channels=16 -c'

  hybridts = 'python3 exp_train_model/main_doce_training.py -s hybridts/transcoder=cnn_pinv+dataset=TEST+classifier=YamNet+epoch=1+learning_rate=-3+kernel_size=5+nb_layers=5+dilation=0+nb_channels=64+prop_logit=50 -c'

  ts = 'python3 exp_train_model/main_doce_training.py -s ts/dataset=TEST+classifier=YamNet+epoch=1+learning_rate=-3 -c'

  if config.plan in ['oracle', 'all']:
    exe = oracle
    os.system(exe)
  if config.plan in ['pinv', 'all']:
    exe = pinv
    os.system(exe)
  if config.plan in ['mlp', 'all']:
    exe = mlp
    os.system(exe)
  if config.plan in ['cnn', 'all']:
    exe = cnn
    os.system(exe)
  if config.plan in ['hybridts', 'all']:
    exe = hybridts
    os.system(exe)
  if config.plan in ['ts', 'all']:
    exe = ts
    os.system(exe)
  #for dataset in datasets:
  #os.system(exe)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=str, default='all', help='Quick test on selected plan')
    config = parser.parse_args()
    main(config)
