import numpy as np
import torch
import torchaudio.transforms as ta_trans

class CommonParams():
    # for STFT
    TARGET_SAMPLE_RATE = 16000
    STFT_WINDOW_LENGTH_SECONDS = 0.025
    STFT_HOP_LENGTH_SECONDS = 0.010

    # for log mel spectrogram
    NUM_MEL_BANDS = 64
    MEL_MIN_HZ = 125
    MEL_MAX_HZ = 7500
    LOG_OFFSET = 0.001  # NOTE 0.01 for vggish, and 0.001 for yamnet

    # convert input audio to segments
    PATCH_WINDOW_IN_SECONDS = 0.96

    # largest feedforward chunk size at test time
    VGGISH_CHUNK_SIZE = 128
    YAMNET_CHUNK_SIZE = 256

    # num of data loading threads
    NUM_LOADERS = 4


class VGGishParams():
    # Copyright 2017 The TensorFlow Authors All Rights Reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # ==============================================================================

    """Global parameters for the VGGish model.
    See vggish_slim.py for more information.
    """

    # Architectural constants.
    NUM_FRAMES = 96  # Frames in input mel-spectrogram patch.
    NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
    EMBEDDING_SIZE = 128  # Size of embedding layer.

    # Hyperparameters used in feature and example generation.
    SAMPLE_RATE = 16000
    STFT_WINDOW_LENGTH_SECONDS = 0.025
    STFT_HOP_LENGTH_SECONDS = 0.010
    NUM_MEL_BINS = NUM_BANDS
    MEL_MIN_HZ = 125
    MEL_MAX_HZ = 7500
    LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.
    EXAMPLE_WINDOW_SECONDS = 0.96  # Each example contains 96 10ms frames
    EXAMPLE_HOP_SECONDS = 0.96     # with zero overlap.

    # Parameters used for embedding postprocessing.
    PCA_EIGEN_VECTORS_NAME = 'pca_eigen_vectors'
    PCA_MEANS_NAME = 'pca_means'
    QUANTIZE_MIN_VAL = -2.0
    QUANTIZE_MAX_VAL = +2.0

    # Hyperparameters used in training.
    INIT_STDDEV = 0.01  # Standard deviation used to initialize weights.
    LEARNING_RATE = 1e-4  # Learning rate for the Adam optimizer.
    ADAM_EPSILON = 1e-8  # Epsilon for the Adam optimizer.

    # Names of ops, tensors, and features.
    INPUT_OP_NAME = 'vggish/input_features'
    INPUT_TENSOR_NAME = INPUT_OP_NAME + ':0'
    OUTPUT_OP_NAME = 'vggish/embedding'
    OUTPUT_TENSOR_NAME = OUTPUT_OP_NAME + ':0'
    AUDIO_EMBEDDING_FEATURE_NAME = 'audio_embedding'


class YAMNetParams():
    # Copyright 2019 The TensorFlow Authors All Rights Reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # ==============================================================================

    """Hyperparameters for YAMNet."""

    # The following hyperparameters (except PATCH_HOP_SECONDS) were used to train YAMNet,
    # so expect some variability in performance if you change these. The patch hop can
    # be changed arbitrarily: a smaller hop should give you more patches from the same
    # clip and possibly better performance at a larger computational cost.
    SAMPLE_RATE = 16000
    STFT_WINDOW_SECONDS = 0.025
    STFT_HOP_SECONDS = 0.010
    MEL_BANDS = 64
    MEL_MIN_HZ = 125
    MEL_MAX_HZ = 7500
    LOG_OFFSET = 0.001
    PATCH_WINDOW_SECONDS = 0.96
    PATCH_HOP_SECONDS = 0.48

    PATCH_FRAMES = int(round(PATCH_WINDOW_SECONDS / STFT_HOP_SECONDS))
    PATCH_BANDS = MEL_BANDS
    NUM_CLASSES = 521
    CONV_PADDING = 'same'
    BATCHNORM_CENTER = True
    BATCHNORM_SCALE = False
    BATCHNORM_EPSILON = 1e-4
    CLASSIFIER_ACTIVATION = 'sigmoid'

    FEATURES_LAYER_NAME = 'features'
    EXAMPLE_PREDICTIONS_LAYER_NAME = 'predictions'


# NOTE for our inference, don't need overlapping windows
# YAMNetParams.PATCH_HOP_SECONDS = YAMNetParams.PATCH_WINDOW_SECONDS
YAMNetParams.PATCH_HOP_SECONDS = 1.0

class WaveformToInput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        audio_sample_rate = CommonParams.TARGET_SAMPLE_RATE
        window_length_samples = int(round(
            audio_sample_rate * CommonParams.STFT_WINDOW_LENGTH_SECONDS
        ))
        hop_length_samples = int(round(
            audio_sample_rate * CommonParams.STFT_HOP_LENGTH_SECONDS
        ))
        fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
        assert window_length_samples == 400
        assert hop_length_samples == 160
        assert fft_length == 512
        
        self.mel_trans_ope = VGGishLogMelSpectrogram(
            CommonParams.TARGET_SAMPLE_RATE, n_fft=fft_length,
            win_length=window_length_samples, hop_length=hop_length_samples,
            f_min=CommonParams.MEL_MIN_HZ,
            f_max=CommonParams.MEL_MAX_HZ,
            n_mels=CommonParams.NUM_MEL_BANDS
        )
        
        # self.power_to_mel = VGGishLogMelSpectrogramFromPower(
        #     CommonParams.TARGET_SAMPLE_RATE, n_fft=fft_length,
        #     win_length=window_length_samples, hop_length=hop_length_samples,
        #     f_min=CommonParams.MEL_MIN_HZ,
        #     f_max=CommonParams.MEL_MAX_HZ,
        #     n_mels=CommonParams.NUM_MEL_BANDS
        # )
        
        self.power_to_mel = VGGishLogMelSpectrogramFromPower(
            32000, n_fft=4096,
            win_length=4096, hop_length=4000,
            f_min=CommonParams.MEL_MIN_HZ,
            f_max=CommonParams.MEL_MAX_HZ,
            n_mels=CommonParams.NUM_MEL_BANDS
        )
        # note that the STFT filtering logic is exactly the same as that of a
        # conv kernel. It is the center of the kernel, not the left edge of the
        # kernel that is aligned at the start of the signal.

    def __call__(self, waveform, sample_rate):
        '''
        Args:
            waveform: torch tsr [num_audio_channels, num_time_steps]
            sample_rate: per second sample rate
        Returns:
            batched torch tsr of shape [N, C, T]
        '''
        x = waveform.mean(axis=0, keepdims=True)  # average over channels
        resampler = ta_trans.Resample(sample_rate, CommonParams.TARGET_SAMPLE_RATE)
        x = resampler(x)
        x = self.mel_trans_ope(x)
        x = x.squeeze(dim=0).T  # # [1, C, T] -> [T, C]

        window_size_in_frames = int(round(
            CommonParams.PATCH_WINDOW_IN_SECONDS / CommonParams.STFT_HOP_LENGTH_SECONDS
        ))
        num_chunks = x.shape[0] // window_size_in_frames

        # reshape into chunks of non-overlapping sliding window
        num_frames_to_use = num_chunks * window_size_in_frames
        x = x[:num_frames_to_use]
        # [num_chunks, 1, window_size, num_freq]
        x = x.reshape(num_chunks, 1, window_size_in_frames, x.shape[-1])
        return x

    def wave_to_mels(self, waveform, sample_rate):
        '''
        Args:
            waveform: torch tsr [num_audio_channels, num_time_steps]
            sample_rate: per second sample rate
        Returns:
            batched torch tsr of shape [N, C, T]
        '''
        x = waveform
        #x = waveform.mean(axis=0, keepdims=True)  # average over channels
        resampler = ta_trans.Resample(sample_rate, CommonParams.TARGET_SAMPLE_RATE)
        x = resampler(x)

        x = self.mel_trans_ope(x)
        x = x.squeeze(dim=0).T  # # [1, C, T] -> [T, C]
        #spectrogram = x.cpu().numpy().copy()

        window_size_in_frames = int(round(
            CommonParams.PATCH_WINDOW_IN_SECONDS / CommonParams.STFT_HOP_LENGTH_SECONDS
        ))

        if YAMNetParams.PATCH_HOP_SECONDS == YAMNetParams.PATCH_WINDOW_SECONDS:
            num_chunks = x.shape[0] // window_size_in_frames

            # reshape into chunks of non-overlapping sliding window
            num_frames_to_use = num_chunks * window_size_in_frames
            x = x[:num_frames_to_use]
            # [num_chunks, 1, window_size, num_freq]
            x = x.reshape(num_chunks, 1, window_size_in_frames, x.shape[-1])
        else:  # generate chunks with custom sliding window length `patch_hop_seconds`
            patch_hop_in_frames = int(round(
                YAMNetParams.PATCH_HOP_SECONDS / CommonParams.STFT_HOP_LENGTH_SECONDS
            ))
            # TODO performance optimization with zero copy
            patch_hop_num_chunks = (x.shape[0] - window_size_in_frames) // patch_hop_in_frames + 1
            num_frames_to_use = window_size_in_frames + (patch_hop_num_chunks - 1) * patch_hop_in_frames
            x = x[:num_frames_to_use]
            x_in_frames = x.reshape(-1, x.shape[-1])
            x_output = np.empty((patch_hop_num_chunks, window_size_in_frames, x.shape[-1]))
            for i in range(patch_hop_num_chunks):
                start_frame = i * patch_hop_in_frames
                x_output[i] = x_in_frames[start_frame: start_frame + window_size_in_frames]
            x = x_output.reshape(patch_hop_num_chunks, 1, window_size_in_frames, x.shape[-1])
            x = torch.tensor(x, dtype=torch.float32)
        return x

    def power_to_mels(self, waveform, sample_rate):
        '''
        Args:
            waveform: torch tsr [num_audio_channels, num_time_steps]
            sample_rate: per second sample rate
        Returns:
            batched torch tsr of shape [N, C, T]
        '''
        
        x = self.power_to_mel(waveform)
        #x = x.squeeze(dim=0).T  # # [1, C, T] -> [T, C]
        #MT: version above deprecated, replaced by own version below [batch, C, T] -> [T, C, batch] or [1, C, T] -> [T, C] if batch is size is 1
        x = x.transpose(0, 2).squeeze(dim=2)

        return x
    

class VGGishLogMelSpectrogram(ta_trans.MelSpectrogram):
    '''
    This is a _log_ mel-spectrogram transform that adheres to the transform
    used by Google's vggish model input processing pipeline
    '''

    def forward(self, waveform):
        r"""
        Args:
            waveform (torch.Tensor): Tensor of audio of dimension (..., time)

        Returns:
            torch.Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time)
        """
        #specgram = np.expand_dims(waveform, axis=0)
        #specgram = waveform.unsqueeze(0)
        specgram = waveform
        
        specgram = self.spectrogram(specgram)

        # NOTE at mel_features.py:98, googlers used np.abs on fft output and
        # as a result, the output is just the norm of spectrogram raised to power 1
        # For torchaudio.MelSpectrogram, however, the default
        # power for its spectrogram is 2.0. Hence we need to sqrt it.
        # I can change the power arg at the constructor level, but I don't
        # want to make the code too dirty
        specgram = specgram ** 0.5

        mel_specgram = self.mel_scale(specgram)
        mel_specgram = torch.log(mel_specgram + CommonParams.LOG_OFFSET)
        return mel_specgram

class VGGishLogMelSpectrogramFromPower(ta_trans.MelSpectrogram):
    '''
    This is a _log_ mel-spectrogram transform that adheres to the transform
    used by Google's vggish model input processing pipeline
    '''

    def forward(self, power):
        r"""
        Args:
            waveform (torch.Tensor): Tensor of audio of dimension (..., time)

        Returns:
            torch.Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time)
        """
        #specgram = self.spectrogram(power)
        # NOTE at mel_features.py:98, googlers used np.abs on fft output and
        # as a result, the output is just the norm of spectrogram raised to power 1
        # For torchaudio.MelSpectrogram, however, the default
        # power for its spectrogram is 2.0. Hence we need to sqrt it.
        # I can change the power arg at the constructor level, but I don't
        # want to make the code too dirty
        
        specgram = power
        #MT: maybe this line must be commented
        #specgram = specgram ** 0.5
        
        mel_specgram = self.mel_scale(specgram)
        mel_specgram = torch.log(mel_specgram + CommonParams.LOG_OFFSET)
        
        return mel_specgram