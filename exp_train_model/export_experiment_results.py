import os
import argparse

def main(config):
    for classifier in ["\'YamNet\'", "\'PANN\'"]:
        if classifier == "\'YamNet\'":
            epoch_cnn = 200
            epoch_self = 200
            epoch_b7 = 200
            epoch_b0 = 200
        if classifier == "\'PANN\'":
            epoch_cnn = 200
            epoch_self = 200
            epoch_b7 = 200
            epoch_b0 = 200
        
        end_command_export = f'-d -e results_training_{classifier}.{config.ext}'
        if config.metric_calculation:
            end_command_l = ['-c', end_command_export]
        else:
            end_command_l = [end_command_export]
        
        for end_command in end_command_l:
            command = f"""python3 exp_train_model/main_doce_training.py -s "
            {{'transcoder':'pinv', 'ts':0, 'step':'metric', 'dataset': 'full', 'classifier':{classifier}, 
            'learning_rate':-99999, 'kernel_size':-99999, 'nb_layers': -99999, 'dilation':-99999, 'nb_channels':-99999, 'prop_logit':-99999, 
            'hidden1':-99999, 'hidden2':-99999, 'epoch':-99999}},
            {{'transcoder':'cnn_pinv', 'ts':0, 'step':'metric', 'dataset': 'full', 'classifier':{classifier}, 
            'learning_rate':-3, 'kernel_size':5, 'nb_layers': 5, 'dilation':1, 'nb_channels':64, 'prop_logit':-99999, 
            'hidden1':-99999, 'hidden2':-99999, 'epoch':{epoch_cnn}}},
            {{'transcoder':'cnn_pinv', 'ts':1, 'step':'metric', 'dataset': 'full', 'classifier':{classifier}, 
            'learning_rate':-3, 'kernel_size':5, 'nb_layers': 5, 'dilation':1, 'nb_channels':64, 'prop_logit':100, 
            'hidden1':-99999, 'hidden2':-99999, 'epoch':{epoch_cnn}}},
            {{'transcoder':'self', 'ts':1, 'step':'metric', 'dataset':'full','classifier':{classifier}, 
            'learning_rate':-4, 'nb_channels':-99999, 'kernel_size':-99999, 'nb_layers':-99999, 'dilation':-99999, 'hidden1':-99999, 'hidden2':-99999, 'epoch':{epoch_self},
            'prop_logit':-99999}}, 
            {{'transcoder':'effnet_b0', 'ts':1, 'step':'metric', 'dataset':'full','classifier':{classifier}, 'learning_rate':-5, 'nb_channels':-99999, 
            'kernel_size':-99999, 'nb_layers':-99999, 'dilation':-99999, 'hidden1':-99999, 'hidden2':-99999, 'epoch':{epoch_b0},
            'prop_logit':-99999}}, 
            {{'transcoder':'effnet_b7', 'ts':1, 'step':'metric', 'dataset':'full','classifier':{classifier}, 'learning_rate':-5, 
            'nb_channels':-99999, 'kernel_size':-99999, 'nb_layers':-99999, 'dilation':-99999, 'hidden1':-99999, 'hidden2':-99999, 
            'epoch':{epoch_b7}, 'prop_logit':-99999}}" {end_command}"""
            os.system(command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate 1s Mels and Third-Octave spectrograms')

    parser.add_argument('--ext', type=str, default='png',
                        help='The extension of the export: png, html, tex')
    parser.add_argument('--metric_calculation', type=bool, default=False)
    
    config = parser.parse_args()
    main(config)

# if export == "all":
#     #mlp
#     command = f"python3 exp_train_model/main_doce_training.py -s mlp/dataset=full+step=metric+epoch=10+classifier=YamNet+transcoder=mlp -d -e mlp_yamnet.{ext}"
#     os.system(command)
#     command = f"python3 exp_train_model/main_doce_training.py -s mlp/dataset=full+step=metric+epoch=10+classifier=PANN+transcoder=mlp -d -e mlp_pann.{ext}"
#     os.system(command)

#     #mlp_pinv
#     command = f"python3 exp_train_model/main_doce_training.py -s mlp/dataset=full+step=metric+epoch=10+classifier=YamNet+transcoder=mlp_pinv -d -e mlp_pinv_yamnet.{ext}"
#     os.system(command)
#     command = f"python3 exp_train_model/main_doce_training.py -s mlp/dataset=full+step=metric+epoch=10+classifier=PANN+transcoder=mlp_pinv -d -e mlp_pinv_pann.{ext}"
#     os.system(command)

#     #cnn_pinv
#     command = f"python3 exp_train_model/main_doce_training.py -s cnn/dataset=full+step=metric+epoch=10+classifier=YamNet+transcoder=cnn_pinv -d -e cnn_pinv_yamnet.{ext}"
#     os.system(command)
#     command = f"python3 exp_train_model/main_doce_training.py -s cnn/dataset=full+step=metric+epoch=10+classifier=PANN+transcoder=cnn_pinv -d -e cnn_pinv_pann.{ext}"
#     os.system(command)

#     #hybridts
#     command = f"python3 exp_train_model/main_doce_training.py -s hybridts/dataset=full+step=metric+epoch=100+classifier=YamNet+transcoder=cnn_pinv -d -e hybridts_yamnet.{ext}"
#     os.system(command)
#     command = f"python3 exp_train_model/main_doce_training.py -s hybridts/dataset=full+step=metric+epoch=100+classifier=PANN+transcoder=cnn_pinv -d -e hybridts_pann.{ext}"
#     os.system(command)

#     #ts
#     command = f"python3 exp_train_model/main_doce_training.py -s ts/dataset=full+step=metric+classifier=YamNet -d '[0, 1,  2]' -e ts_yamnet.{ext}"
#     os.system(command)
#     command = f"python3 exp_train_model/main_doce_training.py -s ts/dataset=full+step=metric+classifier=PANN -d '[0, 1,  2]' -e ts_pann.{ext}"
#     os.system(command)