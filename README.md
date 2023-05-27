# Spectral Transcoder : Using Pretrained Urban Sound Classifiers On Arbitrary Spectral Representations

This repo contains code for the paper: *Spectral Transcoder : Using Pretrained Urban Sound Classifiers On Undersampled Spectral Representations*. Results of the paper can be reproduced looking at section ... . Some audio samples can be generated looking at section ... . Some complementary experiment results are in section ... 

## Setup

The codebase is developed with Python 3.9.15. Install requirements as follows:
```
pip install -r requirements.txt
```
Download the pre-trained models (PANN and Efficient Nets) with the following commands:

```
python3 download_pretrained_models.py
```

Please make sure that you have around 150G of free space in your hard disk to have enough space for the datasets and for running the experiments.

## Paper Results Replication

### Datasets download

Download the [TAU Urban Acoustic Scenes 2020 Mobile](https://dcase.community/challenge2021/task-acoustic-scene-classification) dataset, the [UrbanSound8k](https://urbansounddataset.weebly.com/urbansound8k.html), [SONYC-UST](https://zenodo.org/record/3966543#.ZFtddpHP1kg) dataset using the following commands:

```
python3 download_datasets.py
```

Those datasets must imperatively be downloaded through this command, as the name of the datasets, and the meta data files are slightly modified (duration calculation was wrong in the original dataset). The datasets will be added at the root of the folder. If you want to choose a different path for the datasets (mypath), please use:

```
python3 download_datasets.py --output mypath
```

### Experiment: training models

Execute the following commands to generate third-octave and Mel data from the [TAU Urban Acoustic Scenes 2020 Mobile](https://dcase.community/challenge2021/task-acoustic-scene-classification) dataset. Those data will be used to train and evaluate the models in the subsequent sections. 

```
python3 exp_train_model/create_mel_tho_dataset.py
```

Data will be put at the root of the github folder. In case you want those data to be stored somewhere else (for example in path *yourpath*), please use the following command:

```
python3 exp_train_model/create_mel_tho_dataset.py --output_path yourpath/
```

The experiment plan is developped with [doce](https://doce.readthedocs.io/en/latest/). 
To reproduce only the results shown in the paper, please use the following commands:

```
python3 exp_train_model/launch_experiment/launch_exp_py.py --exp_type restricted
```

To reproduce detailed results, with a comparison between different hyperparameters,
please use the following commands:

```
python3 exp_train_model/launch_experiment/launch_exp_py.py --exp_type detailed
```

Alternatively, the experiment can be launched using slurms (to launch code on jean zay server).
First you need to create your slurms:

```
python3 exp_train_model/launch_experiment/slurms_create.py
```

Then launch the reference experiment using the following command:

```
python3 exp_train_model/launch_experiment/jean_zay_slurm_reference.slurm
```


Wait until the reference plan of the experiment is finished. Then you can launch all the slurm files in different GPUs using the following command:

```
python3 exp_train_model/launch_experiment/launch_exp_slurms.py
```

After the experiment is done computing, you can export the results in a png format in the "results" folder using the following commands:

```
python3 exp_train_model/export_experiment_results.py 
```

The results will show in folder "export" and will be named "results_training_PANN.png" and "results_training_YamNet.png"

If you want to show more detailed results, please read doce documentation, and select 
the plan and the parameters you want to show. Here is an example using the following code,
which will create an export "myresults.png" in the folder "export":

```
python3 exp_train_model/main_doce_training.py -s cnn/dataset=full+classifier=PANN -d -e myresults.png
```

To plot the training curve of the trained model used in the paper, execute the 
following command:

```
python3 exp_train_model/plot_training_curve.py
```


### Experiment: Evaluate models on classification datasets

To evaluate the model on [UrbanSound8k](https://urbansounddataset.weebly.com/urbansound8k.html) and [SONYC-UST](https://zenodo.org/record/3966543#.ZFtddpHP1kg), please execute the following commands to
generate the logit data (classes predictions) from the classifiers and the transcoded classifiers. The
models used for this generation are in the folder "reference_models" 

```
python3 exp_classif_eval/create_logit_dataset.py --audio_dataset_name URBAN-SOUND-8K
python3 exp_classif_eval/create_logit_dataset.py --audio_dataset_name SONYC-UST
```

To launch the experiment and train the additional fully connected layers:

```
python3 exp_classif_eval/main_doce_score.py -s reference/ -c
python3 exp_classif_eval/main_doce_score.py -s deep/ -c
```

Then, you can export the results of the experiment ("results_classif_urbansound8k.png", "results_classif_sonycust.png") in a png format in the "export" folder using the following commands:

```
python3 exp_classif_eval/main_doce_score.py -s "{'dataset':'URBAN-SOUND-8K'}" -d -e results_classif_urbansound8k.png
python3 exp_classif_eval/main_doce_score.py -s "{'dataset':'SONYC-UST'}" -d -e results_classif_sonycust.png
```


## Audio generation

As Mel spectrograms can be inverted with librosa using the feature [mel_to_audio](https://librosa.org/doc/main/generated/librosa.feature.inverse.mel_to_audio.html), we can also invert transcoded Mel spectrograms and thus retrieve audio from third-octave spectrograms. Some audio examples obtained on [freesound](https://freesound.org/) are available in the audio and generated_audio folders. You can try with your own audio files, by putting your wav file ("myfile.wav") in the audio folder and executing this command:

```
python3 generate_audio.py myfile.wav
```

The generated wav files will be placed in the generated_audio folder. It will contain the original file (myfile/myfile_original.wav) the audio file generated from PANN 32ms Mel Spectrogram (myfile/myfile_generated_from_groundtruth_mel.wav), the audio file generated from the PINV transcoder (myfile/myfile_generated_from_pinv.wav), and finally the audoo file generated from the CNN transcoder (myfile/myfile_generated_from_transcoder.wav). 

## Complementary experiment results

To reproduce the spectrogram examples shown in the paper, please use the following code:

```
python3 plot_spectro_dcase2023.py
```

Some training curves, and other complementary results are shown in the results folder. This includes a revised version of F.Gontier et al. aggregation method proposed in the paper [Polyphonic training set synthesis improves self-supervised urban sound classification](https://hal-nantes-universite.archives-ouvertes.fr/hal-03262863/). Instead of taking the top 3, the top 8 classes of SONYC-UST is taken into account for the aggregation. During inference, some of the classes that were considered relevant for each SONYC-UST class are grouped, so that the vector of size 527 becomes a vector of size ... . 


