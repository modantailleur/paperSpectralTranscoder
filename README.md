# Spectral Transcoder : Using Pretrained Urban Sound Classifiers On Arbitrary Spectral Representations

This repo contains code for the paper: *Spectral Transcoder : Using Pretrained Urban Sound Classifiers On Undersampled Spectral Representations*. Results of the paper can be reproduced looking at section 2. Some audio samples can be generated looking at section 3. Some complementary experiment results are in section 4. 

## 1 - Setup

The codebase is developed with Python 3.9.15. Install requirements as follows:
```
pip install -r requirements.txt
```
Download the pre-trained models (PANN and Efficient Nets) with the following commands:

```
python3 download_pretrained_models.py
```

Please make sure that you have around 150G of free space in your hard disk to have enough space for the datasets and for running the experiments.

## 2 - Paper Results Replication

### Datasets download

Download the [TAU Urban Acoustic Scenes 2020 Mobile](https://dcase.community/challenge2021/task-acoustic-scene-classification) dataset, the [UrbanSound8k](https://urbansounddataset.weebly.com/urbansound8k.html), [SONYC-UST](https://zenodo.org/record/3966543#.ZFtddpHP1kg) dataset using the following commands:

```
python3 download_datasets.py
```

Those datasets must imperatively be downloaded through this command, as the name of the datasets, and the meta data files are slightly modified (duration calculation was wrong in the original dataset). The datasets will be added at the root of the folder. If you want to choose a different path for the datasets (mypath), please use:

```
python3 download_datasets.py --output mypath
```

### a) Experiment: training models

Execute the following commands to generate third-octave and Mel data from the [TAU Urban Acoustic Scenes 2020 Mobile](https://dcase.community/challenge2021/task-acoustic-scene-classification) dataset. Those data will be used to train and evaluate the models in the subsequent sections. 

```
python3 exp_train_model/create_mel_tho_dataset.py
```

Data will be put at the root of the github folder. In case you want those data to be stored somewhere else (for example in path *yourpath*), please use the following command:

```
python3 exp_train_model/create_mel_tho_dataset.py --output_path yourpath/
```

The experiment plan is developped with [doce](https://doce.readthedocs.io/en/latest/). 
To reproduce only the results shown in the paper, please use the following commands (about 50h of calculation on a single GPU):

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


### b) Experiment: Evaluate models on classification datasets

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
python3 exp_classif_eval/main_doce_score.py -s deep/dataset=URBAN-SOUND-8K -d [0] -e results_classif_urbansound8k.png
python3 exp_classif_eval/main_doce_score.py -s deep/dataset=SONYC-UST -d [1] -e results_classif_sonycust.png
```


## 3 - Audio generation

As Mel spectrograms can be inverted with librosa using the feature [mel_to_audio](https://librosa.org/doc/main/generated/librosa.feature.inverse.mel_to_audio.html), we can also invert transcoded Mel spectrograms and thus retrieve audio from third-octave spectrograms. Some audio examples obtained on [freesound](https://freesound.org/) are available in the audio and generated_audio folders. You can try with your own audio files, by putting your wav file ("myfile.wav") in the audio folder and executing this command:

```
python3 generate_audio.py myfile.wav
```

The generated wav files will be placed in the generated_audio folder. It will contain the original file (myfile/myfile_original.wav) the audio file generated from PANN 32ms Mel Spectrogram (myfile/myfile_generated_from_groundtruth_mel.wav), the audio file generated from the PINV transcoder (myfile/myfile_generated_from_pinv.wav), and finally the audoo file generated from the CNN transcoder (myfile/myfile_generated_from_transcoder.wav). 

## 4 - Complementary experiment results

Figure below (run plot_spectro_dcase2023.py to replicate the figure) shows the transcoding result with several transcoding algorithm (PINV transcoder, CNN-mels transcoder and CNN-logits transcoder) on a 1s audio excerpt from the evaluation dataset.

<img src="results/spectro_dcase_2023.png" width=400>

Being able to transcode Mel spectrograms into third-octave spectrograms easily would have been convenient for training a deep learning model. This would for example allow an auto-encoding approach. Nevertheless, this task is not obvious, even if there are more Mel bins in Mel spectrograms. Indeed, figures below (run plot_thirdo_mels_bands_repartition.py to replicate the figure) shows the repartition of the Mel spectral bands and the third-octave bands on the frequency axis. As there are more third-octave bands in the lower frequencies (below 1kHz) than there are Mel bands, we cannot easily transcode Mels to third-octaves, particularly in that part of the spectrum. 

<img src="results/thirdo_mels_bands_repartition.png" width=400>

As stated in the paper *Spectral Transcoder : Using Pretrained Urban Sound Classifiers On Undersampled Spectral Representations*, we propose a revised version of F.Gontier et al. aggregation method proposed in the paper [Polyphonic training set synthesis improves self-supervised urban sound classification](https://hal-nantes-universite.archives-ouvertes.fr/hal-03262863/). During inference, some of the classes that were considered relevant for each SONYC-UST class are grouped. The groups made during inference are the one in the files exp_classif_eval/sub_classes_sonyc_ust.xlsx and exp_classif_eval/sub_classes_urbansound8k.xlsx. 

In Gontier et al. 's paper, if in the highest top 3 predictions of YamNet one predicted class among the 527 belongs to one of the meta-classes (traffic, voice, birds in their paper), the meta-class was considered present. Instead of taking the top 3, the top 8 classes of SONYC-UST is taken into account for the aggregation. In our case, we believe that Gontier et al.'s method can result in some false negatives for multi-label classification. For example, if there are is some music in the audio excerpt, it is very likely that the 8 first predicted classes will be related to music (ex: 1:Music, 2:Musical Instrument, 3:Guitar, 4:Pop Music, 5:Drum, 6:Piano, 7:Bass Guitar, 8:Acoustic Guitar), and so the next prediction at position 9 will not be considered present in the audio excerpt (ex: 9: Speech). We propose to group classes together during inference, so that it leads to less false negatives (ex: 1:SONYC-UST Music 2: SONYC-UST HumanVoice, 3: Mosquito etc...). This lead to a higher mAUC than Gontier et al's method for SONYC-UST. The same kind of aggregation is made for UrbanSound8k, but in a more simple multi-class classification paradigm (the meta-class is present if one of its classes has the highest score). 

The results of this method, compared to the method where we add fully connected layers at the output of the pre-trained models (method explained in the paper), are shown in the table below. Where deep is set to 1, it means that fully connected layers are used, where deep is set to 0, it means that the aggregation method mentioned in the previous paragraph is used.

<img src="results/results_classif_sonycust.png" width=400>
<img src="results/results_classif_urbansound8k.png" width=400>

The images above can be replicated using the following commands (after running part 2-b):
```
python3 exp_classif_eval/main_doce_score.py -s "{'dataset':'URBAN-SOUND-8K'}" -d [0] -e results_classif_urbansound8k.png
python3 exp_classif_eval/main_doce_score.py -s "{'dataset':'SONYC-UST'}" -d [1] -e results_classif_sonycust.png
```

