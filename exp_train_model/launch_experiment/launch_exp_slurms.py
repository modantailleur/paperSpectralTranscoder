import os

slurm_dir = "./exp_train_model/slurms"

# Get a list of all Slurm files in the slurm_dir directory
slurm_files = [f for f in os.listdir(slurm_dir) if os.path.isfile(os.path.join(slurm_dir, f)) and f.endswith('.slurm')]

# Iterate over the list of Slurm files and launch each one
for slurm_file in slurm_files:
    command = f"sbatch {os.path.join(slurm_dir, slurm_file)}"
    print(command)
    os.system(command)
