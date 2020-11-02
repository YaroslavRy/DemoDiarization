import torch
import numpy as np
from audio import preprocess_wav, play_wav_file, load_audio_file
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from utils import save_pickle


# !mkdir audio_data/combined
AUDIO_PATH = '/Users/nemo/Downloads/DS_10283_2651/VCTK-Corpus/wav48/'
PATH_TO_SAVE = './audio_data/combined/'
sr = 48000


def combine_utters(utter_list, names, sr):
    label_enc = LabelEncoder()
    label_enc.fit(names)
    res_utters = list(utter_list[0])
    labels = list(label_enc.transform([names[0]])) * utter_list[0].shape[0]
    for i, utter in enumerate(utter_list[1:]):
        res_utters += list(utter)
        labels += list(label_enc.transform([names[i+1]])) * utter.shape[0]
    return np.array(res_utters), np.array(labels)


# Make new wavs combining different speakers wavs
n_speakers = 999
max_speakers = 3
n_spkrs_utters = 1  # number of uterrances for each speaker
speakers_list = [x for x in os.listdir(AUDIO_PATH) if os.path.isdir(AUDIO_PATH+x)]
for i, speaker in enumerate(speakers_list):
    if i == n_speakers:
        break
    speaker_path = AUDIO_PATH + speaker
    if not os.path.isdir(speaker_path):
        continue    
    
    curr_n_speakers = np.random.randint(1, max_speakers+1)
    rndm_spkrs = np.random.choice(speakers_list, curr_n_speakers)
    wavs = []
    combined_labels = []
    times_between = []
    for j, speaker_file in enumerate(os.listdir(speaker_path)):
        if j == n_spkrs_utters:
            break
        speaker_file_path = speaker_path + '/' + speaker_file
        spkr_wav = preprocess_wav(speaker_file_path, sampling_rate=sr, trim_silence=True)
        wavs.append(spkr_wav)
        combined_labels.append(speaker)
        for random_speaker in rndm_spkrs:
            random_speaker_path = AUDIO_PATH + random_speaker
            random_spkr_uttr_name = np.random.choice([x for x in os.listdir(random_speaker_path) if x[-3:] == 'wav'] )
            random_speaker_file_path = random_speaker_path + '/' + random_spkr_uttr_name
            random_spkr_wav = preprocess_wav(random_speaker_file_path, sampling_rate=sr, trim_silence=True)
            wavs.append(random_spkr_wav)
            combined_labels.append(random_speaker)
    combined_utters, labels_encoded = combine_utters(wavs, combined_labels, sr)
    filename = PATH_TO_SAVE + speaker + '_' + str(len(rndm_spkrs))
    filename += '_' + str(np.random.randint(0, 1e+5)) 
    # np.save(filename, list([combined_utters, labels_encoded]))
    save_pickle(list([combined_utters, labels_encoded]), filename)

