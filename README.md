# Spectral Transcoder : Using Pretrained Urban Sound Classifiers On Arbitrary Spectral Representations

This repo contains code for our paper: ... . Results of the paper can be reproduced looking at section ... . Some audio samples can be generated looking at section ... . Some complementary experiment results are in section ... 

## Paper results reproduction

### Environments

The codebase is developed with Python 3.7. Install requirements as follows:
```
pip install -r requirements.txt
```
### Datasets download

Download the [TAU Urban Acoustic Scenes 2020 Mobile](https://dcase.community/challenge2021/task-acoustic-scene-classification) dataset, the [UrbanSound8k](https://urbansounddataset.weebly.com/urbansound8k.html), [SONYC-UST](https://zenodo.org/record/3966543#.ZFtddpHP1kg) dataset using the following commands:


```
command to execute
```

### Third-octave and Mel data generation

Execute the following commands to generate third-octave and Mel data from the [TAU Urban Acoustic Scenes 2020 Mobile](https://dcase.community/challenge2021/task-acoustic-scene-classification) dataset. Those data will be used to train and evaluate the models in the subsequent sections. 

### Launch experiment

The experiment plan is developped with [doce](https://doce.readthedocs.io/en/latest/). Please note that you can launch only partially the experiment following doce tutorial. The full experiment can be launched using the following commands:

```
command to execute
```


## Audio generation

As Mel spectrograms can be inverted with librosa using the feature [mel_to_audio](https://librosa.org/doc/main/generated/librosa.feature.inverse.mel_to_audio.html), we can also invert transcoded Mel spectrograms and thus retrieve audio from third-octave spectrograms. Some audio examples are available in the audio folder. You can try with your own audio files, by putting your wav file in the audio folder and executing this command:

```
command to execute
```

## Complementary experiment results
