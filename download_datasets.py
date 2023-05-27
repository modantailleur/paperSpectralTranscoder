import tarfile
import zipfile
import glob
import subprocess
import shutil
import os
import librosa
import pandas as pd
import argparse

def download_urbansound8k(dir):

    doi = '10.5281/zenodo.1203745'
    output_dir = dir + 'URBAN-SOUND-8K/'
    #for reordering folders
    source_folder = dir + 'URBAN-SOUND-8K/UrbanSound8K/'

    #############
    # GET ZIP FILES FROM ZENODO

    command = ['zenodo_get', doi, '-o', output_dir]
    subprocess.run(command, check=True)


    ############
    # EXTRACT ZIP FILES

    # Get a list of all .tar.gz files in the directory
    tar_files = glob.glob(output_dir + '/*.tar.gz')

    # Extract each .tar.gz file
    for tar_file in tar_files:
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall(output_dir)

    # Move the child "doce" folder to the same level as the parent folder and rename it to "doce_temp"
    shutil.move(source_folder, "temp")

    # Delete the empty source folder
    shutil.rmtree(output_dir)

    # Rename the "doce_temp" folder to "doce"
    os.rename("temp", output_dir)

    ####################
    #REMOVE EVERY "FOLD" FOLDER (every information in on the metadata/UrbanSound8k.csv file)

    # get the path of the audio subfolder
    audio_path = output_dir + "audio/"

    # loop through all directories in audio and move their contents to audio root
    for root, dirs, files in os.walk(audio_path):
        for file in files:
            # get the full path of the file
            file_path = os.path.join(root, file)
            # move the file to the audio root
            if not file.startswith('.'):
                shutil.move(file_path, audio_path)
            
    # loop through all directories in audio and remove them
    for root, dirs, files in os.walk(audio_path):
        for dir in dirs:
            # get the full path of the directory
            dir_path = os.path.join(root, dir)
            # remove the directory and all its contents
            shutil.rmtree(dir_path)

    ##########################
    # RECALCULATE DURATIONS (durations are badly calculated in the original csv file)

    # Set file paths
    csv_path = output_dir + 'metadata/UrbanSound8K.csv'
    audio_dir = output_dir + 'audio'

    # Load CSV into Pandas DataFrame
    df = pd.read_csv(csv_path)

    # Add column for duration
    df['duration'] = 0

    n_file = 0
    len_df = len(df)
    # Loop through each row of the DataFrame
    for index, row in df.iterrows():
        # Get the file path for the audio file
        audio_path = os.path.join(audio_dir, row['slice_file_name'])
        
        # Load the audio file with librosa
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Calculate the duration in seconds
        duration = librosa.get_duration(y=audio, sr=sr)
        
        # Update the duration column in the DataFrame
        df.at[index, 'duration'] = duration

        n_file += 1
        print('\r' + f'{n_file} / {len_df} files have been processed in dataset',end=' ')

    # save the csv with recalculated duration column
    new_csv_path = output_dir + 'metadata/UrbanSound8K_recalculated.csv'
    df.to_csv(new_csv_path, index=False)

    print("Extraction complete.")

def download_sonycust(dir):
    doi = '10.5281/zenodo.3966543'
    output_dir = dir + 'SONYC-UST' + '/'

    #############
    # GET ZIP FILES FROM ZENODO
    command = ['zenodo_get', doi, '-o', output_dir]
    subprocess.run(command, check=True)

    ############
    # EXTRACT ZIP FILES

    # Get a list of all .tar.gz files in the directory
    tar_files = glob.glob(output_dir + '/*.tar.gz')

    # Extract each .tar.gz file
    for tar_file in tar_files:
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall(output_dir)

    ####################
    # REMOVE EVERY "audio-*" FOLDER (and move audio files to a unique "audio" folder)

    # get the path of the audio subfolder
    audio_path = output_dir + "audio/"

    # Create the destination directory if it doesn't exist
    if not os.path.exists(audio_path):
        os.makedirs(audio_path)

    # Get a list of all subdirectories in the source directory
    subdirectories = [subdir for subdir in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, subdir))]
    for subdir in subdirectories:
        if subdir.startswith('audio-'):
            subdirectory_path = os.path.join(output_dir, subdir)
            
            # Get a list of all files in the subdirectory
            files = [file for file in os.listdir(subdirectory_path) if os.path.isfile(os.path.join(subdirectory_path, file))]
            
            # Move each audio file to the destination directory
            for file in files:
                file_path = os.path.join(subdirectory_path, file)
                shutil.move(file_path, audio_path)

            # Delete the subdirectory
            shutil.rmtree(subdirectory_path)

    print("Extraction complete.")

def download_tau(dir):

    doi = '10.5281/zenodo.3819968'
    output_dir = dir + 'TAU-urban-acoustic-scenes-2020-mobile-development' + '/'

    #############
    # GET ZIP FILES FROM ZENODO
    command = ['zenodo_get', doi, '-o', output_dir]
    subprocess.run(command, check=True)

    ############
    # EXTRACT ZIP FILES

    # Get a list of all .zip files in the directory
    zip_files = glob.glob(output_dir + '/*.zip')

    # Extract each .zip file
    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

    ###################
    # MOVE THE TAU DATASET SUBFOLDER TO THE PARENT FOLDER

    subdirectory_path = os.path.join(output_dir, 'TAU-urban-acoustic-scenes-2020-mobile-development')

    # Get the list of files and subdirectories inside the subfolder
    contents = os.listdir(subdirectory_path)

    # Move each item to the parent folder
    for item in contents:
        item_path = os.path.join(subdirectory_path, item)
        destination_path = os.path.join(output_dir, item)
        shutil.move(item_path, destination_path)

    # Remove the now empty subfolder
    os.rmdir(subdirectory_path)

    print("Extraction complete.")


def main(config):
    dataset_list = config.dataset.split(',')

    if "URBAN-SOUND-8K" in dataset_list:
        download_urbansound8k(config.output)

    if "SONYC-UST" in dataset_list:
        download_sonycust(config.output)

    if "TAU" in dataset_list:
        download_tau(config.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--dataset', help='Comma-separated list of datasets to download', default= 'SONYC-UST,URBAN-SOUND-8K,TAU')
    parser.add_argument('-o', '--output', help='Output directory for the datasets', default= './')
    config = parser.parse_args()
    main(config)