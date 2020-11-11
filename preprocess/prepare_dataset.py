#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 01:28:54 2020

@author: nemo
"""

# -*- coding: utf-8 -*-

import numpy as np
# from audio import preprocess_wav, play_wav_file, load_audio_file, plot_spectrogram
import matplotlib.pyplot as plt
import librosa
import librosa.display
plt.style.use('seaborn')
import os
from voice_encoder import VoiceEncoder
plt.style.use('seaborn')
from utils import load_pickle, save_pickle
from scipy import signal
from hparams import sampling_rate, embeddings_slice_length
from tqdm import tqdm
from concurrent import futures
from sys import getsizeof
from audio import preprocess_wav, play_wav_file
import time


# !mkdir './data/'
# !mkdir './data/combined_embeddings/'

sr = sampling_rate


filepath = '../audio_data/combined/p226_3_49895.dat'
# wav, labels = load_pickle(filepath)

# play_wav_file(wav, fs=sr)

# print(wav.shape[0]/sr)

# plot_spectrogram(wav)


def get_embedds_from_wav(file_path, slice_len, mode='dataset'):
    if mode == 'dataset':
        wav, labels = load_pickle(file_path)
    else:
        wav = load_pickle(file_path)
    embedds = []
    labels_emb = []
    slice_len *= sr
    n_slices = int(-np.floor(-wav.shape[0]/slice_len))  # hack to floor to biggest
    prev_ind = 0
    for i in range(n_slices):
        curr_index = int(prev_ind + slice_len)
        emb = encoder.embed_utterance(wav[prev_ind: curr_index], return_partials=False, rate=1.5)
        embedds.append(emb)
        if mode == 'dataset':
            labels_emb.append(int(np.median(labels[prev_ind: curr_index])))
        prev_ind = curr_index
    return (embedds, labels_emb)



def get_embeds(file_path, slice_len):
    wav = preprocess_wav(file_path, sampling_rate=sr)
    embedds = []
    slice_len *= sr
    n_slices = int(-np.floor(-wav.shape[0]/slice_len))  # hack to floor to biggest
    prev_ind = 0
    for i in range(n_slices):
        curr_index = int(prev_ind + slice_len)
        emb = encoder.embed_utterance(wav[prev_ind: curr_index], return_partials=False, rate=1.5)
        embedds.append(emb)
        prev_ind = curr_index
    return embedds


def save_emb_utter(file_path, path_to_save, slice_len, n, mode='dataset'):
    (embedds, labels_emb) = get_embedds_from_wav(file_path=file_path, slice_len=slice_len, mode=mode)
    file_name = file_path.split('/')[-1][:-4]
    save_pickle((embedds, labels_emb), path_to_save + file_name + '.dat')
    print('saved', n ,file_name) if n%10==0 else None
    return (embedds, labels_emb)


encoder = VoiceEncoder('cpu')

COMBINED_UTTERS_PATH = '../audio_data/combined/'
PATH_TO_SAVE = '../data/combined_embeddings/'


emb_slice_len = embeddings_slice_length
skip_first_n = 0
with futures.ThreadPoolExecutor() as executor:
    files = os.listdir(COMBINED_UTTERS_PATH)[skip_first_n:]
    file_pathes = [COMBINED_UTTERS_PATH+file for file in files]
    results = [executor.submit(save_emb_utter, f, PATH_TO_SAVE, emb_slice_len, i) for i, f in enumerate(file_pathes)]


data = []
for future in futures.as_completed(results):
    data.append(future.result())

    
save_pickle(data, 'data/data_embeds.dat')


labels_all = []
for i in os.listdir(PATH_TO_SAVE):
      embedds, labels = load_pickle(PATH_TO_SAVE + i)
      labels_all.append(labels)


plt.hist(np.concatenate(labels_all).flatten())


wav = load_pickle('audio_data/combined/p225_1_592.wav')[0]
wav = preprocess_wav(wav, sampling_rate=sr)
play_wav_file(wav, fs=sr)

# !mkdir data/my_test

start_time = time.time()
emb = get_embeds('audio_data/test.m4a', 0.5)
end_time = time.time()
total_time = end_time - start_time 
print(f'embedds got in {total_time:.2f} seconds')


save_pickle(emb, 'data/my_test/test_voice_embeddings.dat')


# =============================================================================
# 
# embedds = []
# labels_emb = []
# slice_len = sr * 0.5
# prev_ind = 0
# for i in range(int(-np.floor(-wav.shape[0]/slice_len))):
#     curr_index = int(prev_ind + slice_len)
#     emb = encoder.embed_utterance(wav[prev_ind: curr_index], return_partials=False, rate=1.5)
#     embedds.append(emb)
#     labels_emb.append(int(np.median(labels[prev_ind: curr_index])))
#     prev_ind = curr_index
# 
# plt.plot(np.array(embedds).flatten())
# 
# plt.plot(embedds)
# 
# 
# mean_emb = np.std(embedds, axis=1)
# plt.plot((mean_emb - np.min(mean_emb))/(np.max(mean_emb) - np.min(mean_emb)))
# plt.plot(labels_emb)
# 
# for i in labels_emb:
#     plt.hist(i)
# 
# 
# f, t, Sxx = signal.spectrogram(wav, sr, )
# S_db = librosa.amplitude_to_db(np.abs(Sxx), ref=np.max)
# plt.pcolormesh(t, f, S_db, shading='nearest', cmap=plt.cm.viridis)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()
# 
# 
# plt.specgram(wav, Fs=sr, cmap=plt.cm.inferno)
# plt.grid()
# 
# 
# plt.plot(labels)
# 
# =============================================================================
