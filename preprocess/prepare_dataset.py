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
from utils import load_pickle
from scipy import signal
from hparams import sampling_rate


PATH_TO_SAVE = './audio_data/combined/'
AUDIO_PATH = '/Users/nemo/Downloads/DS_10283_2651/VCTK-Corpus/wav48/'
sr = sampling_rate

print(os.listdir(AUDIO_PATH + 'p301')) 

filepath = AUDIO_PATH + 'p304/p304_232.wav'
a = preprocess_wav(filepath, sampling_rate=sr)

filepath = './audio_data/combined/p228_2_11960'

wav, labels = load_pickle(filepath)

play_wav_file(wav, fs=sr)

print(a.shape[0]/sr)

plot_spectrogram(wav)


encoder = VoiceEncoder('cpu')
_, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=1)
wav_splits


embedds = []
labels_emb = []
slice_len = sr * 0.5
prev_ind = 0
for i in range(int(-np.floor(-wav.shape[0]/slice_len))):
    curr_index = int(prev_ind + slice_len)
    emb = encoder.embed_utterance(wav[prev_ind: curr_index], return_partials=False, rate=1.5)
    embedds.append(emb)
    labels_emb.append(int(np.median(labels[prev_ind: curr_index])))
    prev_ind = curr_index

plt.plot(np.array(embedds).flatten())

plt.plot(embedds)

mean_emb = np.mean(embedds, axis=1)
plt.plot((mean_emb - np.min(mean_emb))/(np.max(mean_emb) - np.min(mean_emb)))
plt.plot(labels_emb)


for i in labels_emb:
    plt.hist(i)


f, t, Sxx = signal.spectrogram(wav, sr, )
S_db = librosa.amplitude_to_db(np.abs(Sxx), ref=np.max)
plt.pcolormesh(t, f, S_db, shading='nearest', cmap=plt.cm.viridis)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


plt.specgram(wav, Fs=sr, cmap=plt.cm.inferno)
plt.grid()


plt.plot(labels)

