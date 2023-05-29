import librosa
import matplotlib.pyplot as plt
import utils.bands_transform as bt
import numpy as np

mel_filter_yamnet = librosa.filters.mel(sr=32000, n_fft=8096, n_mels=64, fmin=125, fmax=7500, htk=False, norm=None)
mel_filter_pann = librosa.filters.mel(sr=32000, n_fft=8096, n_mels=64, fmin=50, fmax=14000, htk=False, norm=None)
tho_filter = bt.ThirdOctaveTransform(32000, 8096, 4000).tho_basis
scaling_factor = (32000 / 8096)/1000

mel_filter_yamnet_index = np.argmax(mel_filter_yamnet, axis=1)*scaling_factor
mel_filter_pann_index = np.argmax(mel_filter_pann, axis=1)*scaling_factor
tho_filter_index = np.argmax(tho_filter, axis=1)*scaling_factor
plt.xlabel('frequency bin')
plt.ylabel('bin center frequency (kHz)')
plt.scatter(np.arange(1,len(mel_filter_yamnet_index)+1), mel_filter_yamnet_index, marker="+")
plt.scatter(np.arange(1,len(mel_filter_pann_index)+1), mel_filter_pann_index, marker="+")
plt.scatter(np.arange(1,len(tho_filter_index)+1), tho_filter_index, marker="+")
plt.ylim(0, 14)
plt.xlim(0,64)
plt.show()