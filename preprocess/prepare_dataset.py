#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 01:28:54 2020

@author: nemo
"""

# -*- coding: utf-8 -*-

import numpy as np
from audio import preprocess_wav, play_wav_file, load_audio_file, plot_spectrogram
import matplotlib.pyplot as plt
import librosa
import librosa.display
plt.style.use('seaborn')
import os
from voice_encoder import VoiceEncoder
plt.style.use('seaborn')
from utils import load_pickle, save_pickle
from scipy import signal
from hparams import sampling_rate
from tqdm import tqdm
from concurrent import futures


# !mkdir './data/'
# !mkdir './data/combined_embeddings/'

sr = sampling_rate


filepath = '../audio_data/combined/p226_3_49895.dat'
# wav, labels = load_pickle(filepath)

# play_wav_file(wav, fs=sr)

# print(wav.shape[0]/sr)

# plot_spectrogram(wav)


def get_embedds_from_wav(file_path, slice_len):
    wav, labels = load_pickle(file_path)
    embedds = []
    labels_emb = []
    slice_len *= sr
    n_slices = int(-np.floor(-wav.shape[0]/slice_len))  # hack to floor to biggest
    prev_ind = 0
    for i in range(n_slices):
        curr_index = int(prev_ind + slice_len)
        emb = encoder.embed_utterance(wav[prev_ind: curr_index], return_partials=False, rate=1.5)
        embedds.append(emb)
        labels_emb.append(int(np.median(labels[prev_ind: curr_index])))
        prev_ind = curr_index
    return (embedds, labels_emb)


def prepare_dataset(path_combined_utters, path_to_save, slice_len=0.1):
    utters_list = os.listdir(path_combined_utters)
    for filename in tqdm(utters_list):
        (embedds, labels_emb) = get_embedds_from_wav(file_path=path_combined_utters + filename, slice_len=slice_len)
        save_pickle((embedds, labels_emb), path_to_save + filename[:-4] + '.dat')


def save_emb_utter(file_path, path_to_save, slice_len):
    (embedds, labels_emb) = get_embedds_from_wav(file_path=file_path, slice_len=slice_len)
    save_pickle((embedds, labels_emb), path_to_save + file_path[:-4] + '.dat')


encoder = VoiceEncoder('cpu')

COMBINED_UTTERS_PATH = '../audio_data/combined/'
PATH_TO_SAVE = '../data/combined_embeddings/'

# prepare_dataset(path_combined_utters=COMBINED_UTTERS_PATH, path_to_save=PATH_TO_SAVE, slice_len=0.1)


with futures.ThreadPoolExecutor(max_workers=10) as executor:
    files = os.listdir(COMBINED_UTTERS_PATH)
    file_pathes = [COMBINED_UTTERS_PATH+file for file in files]
    results = [executor.submit(save_emb_utter, f, PATH_TO_SAVE, 0.1) for f in file_pathes]


labels_all = []
for i in os.listdir(PATH_TO_SAVE):
     embedds, labels = load_pickle(PATH_TO_SAVE + i)
     labels_all.append(labels)


plt.hist(np.concatenate(labels_all).flatten())


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



for i in range(10):
    for j in range(5):
        print(1)
    print(2)