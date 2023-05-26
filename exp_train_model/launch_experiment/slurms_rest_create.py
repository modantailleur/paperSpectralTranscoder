import os
import shutil
import send2trash

def send_folder_to_trash(folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            send2trash.send2trash(item_path)
        elif os.path.isdir(item_path):
            send_folder_to_trash(item_path)
            send2trash.send2trash(item_path)

send_folder_to_trash("./slurms/")
# Specify the original and new file paths
original_file = os.path.join("jean_zay_slurm_reference.slurm")

step = 'step=eval+'

user_outputs = ['cnn/dataset=full+classifier=PANN+epoch=10+learning_rate=-3+dilation=1+nb_layers=5+kernel_size=5+nb_channels=64 -c', \
		'ts/dataset=full+transcoder=effnet_b0+classifier=PANN+epoch=50+learning_rate=-5 -c', \
		'ts/dataset=full+transcoder=effnet_b7+classifier=PANN+epoch=50+learning_rate=-5 -c', \
		'ts/dataset=full+transcoder=self+classifier=PANN+epoch=50+learning_rate=-4 -c', \
		'hybridts/dataset=full+classifier=PANN+epoch=50+learning_rate=-4+dilation=1+prop_logit=100 -c', \
		
		'cnn/dataset=full+classifier=YamNet+epoch=10+learning_rate=-3+dilation=1+nb_layers=5+kernel_size=5+nb_channels=64 -c', \
		'ts/dataset=full+transcoder=effnet_b0+classifier=YamNet+epoch=50+learning_rate=-5 -c', \
		'ts/dataset=full+transcoder=effnet_b7+classifier=YamNet+epoch=50+learning_rate=-5 -c', \
		'ts/dataset=full+transcoder=self+classifier=YamNet+epoch=50+learning_rate=-4 -c', \
		'hybridts/dataset=full+classifier=YamNet+epoch=50+learning_rate=-4+dilation=1+prop_logit=100 -c']


for k, user_output in enumerate(user_outputs):
    new_file = os.path.join("slurms", "jean_zay_slurm_" + str(k)+".slurm")

    # Copy the original file
    shutil.copyfile(original_file, new_file)

    # Modify the last line of the new file
    # user_output = 'desired_output'
    with open(new_file, 'r') as f:
        content = f.readlines()

    content[-1] = "python3 exp_train_model/main_doce_training.py -s " + user_output + "\n"

    with open(new_file, 'w') as f:
        f.writelines(content)
