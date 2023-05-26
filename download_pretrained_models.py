import subprocess
import argparse

####################
#DOWNLOAD PANN MODEL
output_dir = "./pann"
file_name = "ResNet38_mAP=0.434.pth"
dl_link = "https://zenodo.org//record/3987831/files/ResNet38_mAP%3D0.434.pth?download=1"

# Execute the wget command
command = f'wget "{dl_link}" -O {output_dir}/{file_name}'
subprocess.run(command, shell=True)

###################
#DOWNLOAD EFFICIENT NETS
output_dir = "./efficient_net"

file_name = "efficientnet-b0-355c32eb.pth"
dl_link = "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth"
command = f'wget "{dl_link}" -O {output_dir}/{file_name}'
subprocess.run(command, shell=True)

file_name = "efficientnet-b7-dcc49843.pth"
dl_link = "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth"
command = f'wget "{dl_link}" -O {output_dir}/{file_name}'
subprocess.run(command, shell=True)