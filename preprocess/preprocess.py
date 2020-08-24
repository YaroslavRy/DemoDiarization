# -*- coding: utf-8 -*-

import torch
import numpy as np
from audio import preprocess_wav, play_wav_file, load_audio_file
import matplotlib.pyplot as plt
import os


# !mkdir audio_data/combined
AUDIO_PATH = '/Users/nemo/Downloads/DS_10283_2651/VCTK-Corpus/wav48/'
PATH_TO_SAVE = './audio_data/combined/'
sr = 48000


def combine_utters(utter_list, sr, time_between):
    res = list(utter_list[0])
    for i, utter in enumerate(utter_list[1:]):
        silence_between = np.random.normal(0, 0.001, int(time_between * sr))
        res += list(silence_between)
        res += list(utter)
    return np.array(res)


def get_embeddings():
    return


def save_embeddings():
    return


# Make new wavs combining different spkears wavs
n_speakers = 2
max_speakers = 3 
n_spkrs_utters = 5  # number of uterrances for each speaker
speakers_list = os.listdir(AUDIO_PATH)
for i, speaker in enumerate(speakers_list):
    if i >= n_speakers:
        break
    speaker_path = AUDIO_PATH + speaker
    if not os.path.isdir(speaker_path):
        continue    
    
    curr_n_speakers = np.random.randint(1, max_speakers+1)
    rndm_spkrs = np.random.choice(speakers_list, curr_n_speakers)
    combined_utters = []
    for random_speaker in rndm_spkrs:
        random_speaker_path = AUDIO_PATH + random_speaker
        random_spkr_uttr_name = np.random.choice(os.listdir(random_speaker_path))
        random_speaker_file_path = random_speaker_path + '/' + random_spkr_uttr_name
        for j, speaker_file in enumerate(os.listdir(speaker_path)):
            if j>n_spkrs_utters:
                break
            speaker_file_path = speaker_path + '/' + speaker_file
            spkr_wav = preprocess_wav(speaker_file_path, sampling_rate=sr, trim_silence=False)
            random_spkr_wav = preprocess_wav(random_speaker_file_path, sampling_rate=sr, trim_silence=False)
            time_between = np.random.randint(0, 3, 1)[0]
            combined_utters += list(combine_utters([spkr_wav, random_spkr_wav], sr, time_between))
    filename = PATH_TO_SAVE+speaker + '_' + str(len(rndm_spkrs))
    np.save(filename, combined_utters)

