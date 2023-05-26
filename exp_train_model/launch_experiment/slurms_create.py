import os
import shutil
import send2trash
import argparse

TRAIN_EXP_PATH = "./exp_train_model/"

def send_folder_to_trash(folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            send2trash.send2trash(item_path)
        elif os.path.isdir(item_path):
            send_folder_to_trash(item_path)
            send2trash.send2trash(item_path)

def main(config):
    send_folder_to_trash(TRAIN_EXP_PATH+"slurms")
    # Specify the original and new file paths
    original_file = os.path.join(TRAIN_EXP_PATH+"launch_experiment/jean_zay_slurm_reference.slurm")

    if config.experiment_size == "detailed":
        user_outputs = [
                    'mlp/transcoder=mlp+dataset=full+classifier=YamNet+epoch=200+hidden1=100+hidden2=3000+learning_rate=-3 -c', \
                    'mlp/transcoder=mlp+dataset=full+classifier=PANN+epoch=200+hidden1=100+hidden2=3000+learning_rate=-3 -c', \
                    'mlp/transcoder=mlp_pinv+dataset=full+classifier=YamNet+epoch=200+hidden1=100+hidden2=3000+learning_rate=-3 -c', \
                    'mlp/transcoder=mlp_pinv+dataset=full+classifier=PANN+epoch=200+hidden1=100+hidden2=3000+learning_rate=-3 -c', \
                    'cnn/dataset=full+classifier=PANN+epoch=200+nb_channels=64+kernel_size=5+nb_layers=5+dilation=1+learning_rate=-3 -c', \
                    'cnn/dataset=full+classifier=YamNet+epoch=200+nb_channels=64+kernel_size=5+nb_layers=5+dilation=1+learning_rate=-3 -c', \
                    
                    'hybridts/dataset=full+classifier=YamNet+epoch=200+prop_logit=100+kernel_size=5+nb_layers=5+dilation=1+learning_rate=-3 -c', \
                    'hybridts/dataset=full+classifier=PANN+epoch=200+prop_logit=100+kernel_size=5+nb_layers=5+dilation=1+learning_rate=-3 -c', \
                    'hybridts/dataset=full+classifier=YamNet+epoch=200+prop_logit=100+kernel_size=5+nb_layers=5+dilation=1+learning_rate=-4 -c', \
                    'hybridts/dataset=full+classifier=PANN+epoch=200+prop_logit=100+kernel_size=5+nb_layers=5+dilation=1+learning_rate=-4 -c', \
                    'hybridts/dataset=full+classifier=YamNet+epoch=200+prop_logit=100+kernel_size=5+nb_layers=5+dilation=1+learning_rate=-5 -c', \
                    'hybridts/dataset=full+classifier=PANN+epoch=200+prop_logit=100+kernel_size=5+nb_layers=5+dilation=1+learning_rate=-5 -c', \
                    
                    'ts/dataset=full+transcoder=self+classifier=YamNet+epoch=200+learning_rate=-3 -c', \
                    'ts/dataset=full+transcoder=self+classifier=PANN+epoch=200+learning_rate=-3 -c', \
                    'ts/dataset=full+transcoder=self+classifier=YamNet+epoch=200+learning_rate=-4 -c', \
                    'ts/dataset=full+transcoder=self+classifier=PANN+epoch=200+learning_rate=-4 -c', \
                    'ts/dataset=full+transcoder=self+classifier=YamNet+epoch=200+learning_rate=-5 -c', \
                    'ts/dataset=full+transcoder=self+classifier=PANN+epoch=200+learning_rate=-5 -c', \
                    
                    'ts/dataset=full+transcoder=effnet_b0+classifier=YamNet+epoch=200+learning_rate=-3 -c', \
                    'ts/dataset=full+transcoder=effnet_b0+classifier=PANN+epoch=200+learning_rate=-3 -c', \
                    'ts/dataset=full+transcoder=effnet_b0+classifier=PANN+epoch=200+learning_rate=-4 -c',
                    'ts/dataset=full+transcoder=effnet_b0+classifier=YamNet+epoch=200+learning_rate=-4 -c', \
                    'ts/dataset=full+transcoder=effnet_b0+classifier=YamNet+epoch=200+learning_rate=-5 -c', \
                    'ts/dataset=full+transcoder=effnet_b0+classifier=PANN+epoch=200+learning_rate=-5 -c'

                    'ts/dataset=full+transcoder=effnet_b7+classifier=YamNet+epoch=200+learning_rate=-3 -c', \
                    'ts/dataset=full+transcoder=effnet_b7+classifier=PANN+epoch=200+learning_rate=-3 -c', \
                    'ts/dataset=full+transcoder=effnet_b7+classifier=YamNet+epoch=200+learning_rate=-4 -c',
                    'ts/dataset=full+transcoder=effnet_b7+classifier=PANN+epoch=200+learning_rate=-4 -c', \
                    'ts/dataset=full+transcoder=effnet_b7+classifier=YamNet+epoch=200+learning_rate=-5 -c', \
                    'ts/dataset=full+transcoder=effnet_b7+classifier=PANN+epoch=200+learning_rate=-5 -c']

    if config.experiment_size == "restricted":
        user_outputs = ['cnn/dataset=full+classifier=PANN+epoch=200+nb_channels=64+kernel_size=5+nb_layers=5+dilation=1+learning_rate=-3 -c', \
                    'cnn/dataset=full+classifier=YamNet+epoch=200+nb_channels=64+kernel_size=5+nb_layers=5+dilation=1+learning_rate=-3 -c', \
                    
                    'hybridts/dataset=full+classifier=YamNet+epoch=200+prop_logit=100+kernel_size=5+nb_layers=5+dilation=1+learning_rate=-3 -c', \
                    'hybridts/dataset=full+classifier=PANN+epoch=200+prop_logit=100+kernel_size=5+nb_layers=5+dilation=1+learning_rate=-3 -c', \
                    
                    'ts/dataset=full+transcoder=self+classifier=YamNet+epoch=200+learning_rate=-4 -c', \
                    'ts/dataset=full+transcoder=self+classifier=PANN+epoch=200+learning_rate=-4 -c', \
                    
                    'ts/dataset=full+transcoder=effnet_b0+classifier=YamNet+epoch=200+learning_rate=-5 -c', \
                    'ts/dataset=full+transcoder=effnet_b0+classifier=PANN+epoch=200+learning_rate=-5 -c'

                    'ts/dataset=full+transcoder=effnet_b7+classifier=YamNet+epoch=200+learning_rate=-5 -c', \
                    'ts/dataset=full+transcoder=effnet_b7+classifier=PANN+epoch=200+learning_rate=-5 -c']


    for k, user_output in enumerate(user_outputs):
        new_file = os.path.join(TRAIN_EXP_PATH+"slurms", "jean_zay_slurm_" + str(k)+".slurm")

        # Copy the original file
        shutil.copyfile(original_file, new_file)

        # Modify the last line of the new file
        # user_output = 'desired_output'
        with open(new_file, 'r') as f:
            content = f.readlines()

        content[-1] = "python3 exp_train_model/main_doce_training.py -s " + user_output + "\n"

        with open(new_file, 'w') as f:
            f.writelines(content)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create the slurms file that will be launched with launch_exp_slurms.py. The slurm files will be located in ./slurms/')

    parser.add_argument('--experiment_size', type=str, default="restricted",
                        help='The experiment size to run. If restricted, only launches the experiment that are shown in the SpectralTranscoder paper. \
                                If detailed, launches a detail experiment, comparing models with different hyperparameter values.')
    config = parser.parse_args()
    main(config)