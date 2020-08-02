from audio import preprocess_wav, play_wav_file
from voice_encoder import VoiceEncoder
from demo_utils import *
from pathlib import Path
from time import perf_counter as timer
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
from sklearn.decomposition import PCA


# DEMO 02: we'll show how this similarity measure can be used to perform speaker diarization
# (telling who is speaking when in a recording).


sr = 48000

## Get reference audios
# Load the interview audio from disk
# Source for the interview: https://www.youtube.com/watch?v=X2zqiX6yL3I
wav_fpath = Path("audio_data", "X2zqiX6yL3I.mp3")
wav_fpath = Path("audio_data", "two_voices.m4a")
wav_fpath = Path("audio_data", "two_voices_b.m4a")
wav_fpath = Path("audio_data", "two_voices_mark.m4a")

first_n_seconds = 10
wav = preprocess_wav(wav_fpath, sampling_rate=sr, trim_silence=False)[:sr*first_n_seconds]
play_wav_file(wav, fs=sr)

# Cut some segments from single speakers as reference audio
segments = [[0, 5.5], [6.5, 12], [17, 25]]
speaker_names = ["Kyle Gass", "Sean Evans", "Jack Black"]
speaker_wavs = [wav[int(s[0] * sampling_rate):int(s[1]) * sampling_rate] for s in segments]

    
## Compare speaker embeds to the continuous embedding of the interview
# Derive a continuous embedding of the interview. We put a rate of 16, meaning that an 
# embedding is generated every 0.0625 seconds. It is good to have a higher rate for speaker 
# diarization, but it is not so useful for when you only need a summary embedding of the 
# entire utterance. A rate of 2 would have been enough, but 16 is nice for the sake of the 
# demonstration. 
# We'll exceptionally force to run this on CPU, because it uses a lot of RAM and most GPUs 
# won't have enough. There's a speed drawback, but it remains reasonable.
encoder = VoiceEncoder("cpu")
print("Running the continuous embedding on cpu, this might take a while...")
start = timer()
_, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=2)
print('Embeddings got in {} seconds'.format(timer() - start))


# Get the continuous similarity for every speaker. It amounts to a dot product between the 
# embedding of the speaker and the continuous embedding of the interview
speaker_embeds = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]
similarity_dict = {name: cont_embeds @ speaker_embed for name, speaker_embed in 
                   zip(speaker_names, speaker_embeds)}

print('Embedding output shape:', cont_embeds.shape)


for speaker_wav in speaker_wavs:
    plt.plot(speaker_wav)
plt.show()


plt.plot(cont_embeds)

avg_emb_vects = np.mean(cont_embeds, axis=1)
pca = PCA(n_components=1)

emb_vects_reduced = pca.fit_transform(cont_embeds).flatten()

plt.plot(wav)
plt.plot(avg_emb_vects)
plt.xticks(np.arange(0, wav.shape[0], sr))
plt.show()


size = emb_vects_reduced.shape[0]
xloc = np.arange(size)
new_size = wav.shape[0]
new_xloc = np.linspace(0, size, new_size)
stretched_avg_emb_vects = np.interp(new_xloc, xloc, emb_vects_reduced)
plt.plot(stretched_avg_emb_vects)
plt.show()
plt.plot(wav)
plt.show()
play_wav_file(wav, fs=sr)

for k, v in similarity_dict.items():
    plt.plot(v, label=k)
