import soundfile as sf
import os
import matplotlib.pyplot as plt
import params as yamnet_params
import yamnet as yamnet_model
import tensorflow as tf
import numpy as np
from tqdm import tqdm

def run(dataMel):
    params = yamnet_params.Params(sample_rate=44100, patch_hop_seconds=.128)
    
    # Set up the YAMNet model.
    yamnet = yamnet_model.yamnet_frames_model_logmel(params, dm_input=tf.convert_to_tensor(dataMel[0,:,:], dtype=tf.float32))
    yamnet.load_weights(os.path.dirname(__file__)+'/yamnet.h5')
    
    scores = np.zeros((dataMel.shape[0], 339, 521))
    embeddings = np.zeros((dataMel.shape[0], 339, 1024))
    for k in tqdm(range(dataMel.shape[0])):
        s, e, spec = yamnet(np.squeeze(dataMel[k,:,:]))
        scores[k, :, :] = s.numpy()
        embeddings[k, :, :] = e.numpy()
    return scores, embeddings

def score2presence(scores, annotationType='short'):
    presence = np.zeros((scores.shape[0], scores.shape[1], 3))
    top_classes = 3
    # long
    if annotationType == 'long':
      traffic = np.array([300, 307, 308, 309, 310, 315, 320, 321])
      voice = np.arange(22) # 22
      bird = 106+np.arange(11)
    # short
    if annotationType == 'short':
      traffic = np.array([300, 308, 310, 315, 320, 321])
      voice = np.arange(4) # 22
      bird = 106+np.arange(10)

    for k in range(30):
      presence[k, :, 0] = np.max(np.isin(np.argsort(scores[k, :, :])[:, -top_classes:], traffic), axis = 1).astype(int)
      presence[k, :, 1] = np.max(np.isin(np.argsort(scores[k, :, :])[:, -top_classes:], voice), axis = 1).astype(int)
      presence[k, :, 2] = np.max(np.isin(np.argsort(scores[k, :, :])[:, -top_classes:], bird), axis = 1).astype(int)
    return presence
