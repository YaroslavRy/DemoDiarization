# -*- coding: utf-8 -*-

import numpy as np
from audio import preprocess_wav, play_wav_file, load_audio_file
import matplotlib.pyplot as plt
import os
from voice_encoder import VoiceEncoder
plt.style.use('seaborn')


PATH_TO_SAVE = './audio_data/combined/'
AUDIO_PATH = '/Users/nemo/Downloads/DS_10283_2651/VCTK-Corpus/wav48/'
sr = 48000

os.listdir(AUDIO_PATH + 'p304/')


a = preprocess_wav(AUDIO_PATH + 'p304/' + 'p304_233.wav', sampling_rate=sr)
b = preprocess_wav(AUDIO_PATH + 'p304/' + 'p304_228.wav', sampling_rate=sr)
c = preprocess_wav(AUDIO_PATH + 'p303/' + 'p303_058.wav', sampling_rate=sr)

play_wav_file(a, sr)

encoder = VoiceEncoder("cpu")


n = 15750
e_a, emb, wav_splits = encoder.embed_utterance(a[:n], return_partials=True)
print('emb shape {}, another shape {}'.format(emb.shape, e_a.shape))

a.shape[0]/n

e_a, embed_a, wav_splits = encoder.embed_utterance(a, return_partials=True, min_coverage=1)
e_b, embed_b, wav_splits = encoder.embed_utterance(b, return_partials=True, min_coverage=1)
e_c, embed_c, wav_splits = encoder.embed_utterance(c, return_partials=True, min_coverage=1)

embed_a.shape
embed_b.shape
embed_c.shape

embed_b = embed_b[:embed_a.shape[0]:, ]

plt.plot(a)
plt.plot(b)
plt.plot(embed_a)
plt.plot(embed_b)


plt.plot(np.mean(np.dot(embed_a, embed_b.T),axis=1))
plt.plot(np.mean(np.dot(embed_a, embed_c.T),axis=1))


plt.plot(e_a)
plt.plot(e_b)
plt.plot(e_c)

np.inner(e_a, e_c)


# feed every n seconds of utter to NN
l = 1   # length of utter to feed
arr = []
for i in range(int(len(a)/sr)+1):
    curr_part = b[i*sr:i+l*sr]
    e_a, embed_a, wav_splits = encoder.embed_utterance(curr_part, 
                                                       return_partials=True,
                                                       min_coverage=1)
    arr.append(e_a)

len(arr)


e_a, e_c = arr[1], arr[2]

plt.plot(e_a)
plt.plot(e_b)


