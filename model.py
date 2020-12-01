from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Input, TimeDistributed, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import load_pickle, save_pickle
from audio import play_wav_file, preprocess_wav
from hparams import sampling_rate
import os
plt.style.use('seaborn')


def build_model(timesteps, n_features, lr, gru_units, fc_units, output_shape):
    K.clear_session()
    model = Sequential()
    model.add(GRU(units=gru_units, input_shape=(timesteps, n_features,), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(units=gru_units//2, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(TimeDistributed(Dense(units=output_shape, activation='softmax')))
    model.compile(optimizer=Adam(lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


class DiarizationModel:
    def __init__(self):
        self.model = load_model('models/model_diarization')


if __name__ == '__main__':
    
    
    data = load_pickle('data/data_embeds.dat')
    data = np.array(data)
    X, y = data[:, 0], data[:, 1]
    
    
    ls_x = []
    ls_y = []
    for d in data:
        ls_x.append(len(d[0]))
        ls_y.append(len(d[1]))
    
    plt.hist(ls_x)
    plt.hist(ls_y, alpha=0.5)
    
    
    max_len = 50
    
    
    X_padded = pad_sequences(X, value=0, dtype=float, padding='post', maxlen=max_len)
    y_padded = pad_sequences(y, value=4, padding='post', maxlen=max_len)
    print(X_padded.shape)
    
    n_unique_targets = np.unique(y_padded).shape[0]
    
    
    # TODO: move to config
    lr = 0.001
    batch_size = 32
    gru_units = 256
    input_dim = max_len  # represents a sequence length.
    n_features = 256  # number of featues, vectors representing voice characteristics
    output_shape = n_unique_targets
    fc_units = 32
    
    
    model = build_model(input_dim, n_features, lr=lr, gru_units=gru_units, fc_units=fc_units, output_shape=output_shape)
    model.summary()
    

    
    model.fit(X_padded, y_padded, batch_size=batch_size, epochs=1000, validation_split=0.15)
    
    
    plt.plot(np.mean(X_padded[1], axis=1))
    
    plt.plot(y_padded[1])
    
    
    # model.save('models/model_diarization3')
    model = load_model('models/model_diarization3')
    
    
    # =============================================================================
    # Custom test
    # =============================================================================
    
    sr = sampling_rate
    
    emb_test = load_pickle('data/my_test/test_voice_embeddings.dat')
    
    wav = load_pickle('audio_data/my_test/p236_3_9730.wav')[0]
    
    wav = preprocess_wav('audio_data/test.m4a', sampling_rate=sampling_rate)
    
    play_wav_file(wav, fs=sr)
    
    emb_test_padded = pad_sequences(np.array([emb_test]), dtype=float, padding='post', maxlen=max_len)
    emb_test_padded.shape
    
    preds = model.predict(emb_test_padded)
    
    
    plt.plot(np.argmax(preds[0], axis=1))
    plt.show()
    
    play_wav_file(wav, fs=sr)
    
    
    for i in range(5):
        plt.plot(preds[0].T[i], label=i)
    plt.legend()
    
    
    rnd_data = np.random.randint(0, 100, size=(1, 46, 256))
    rnd_data = rnd_data / np.max(rnd_data)
    
    preds = model.predict(rnd_data)
    
    
    plt.plot(np.mean(preds[0], axis=1))
    plt.show()
    
