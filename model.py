from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Input
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import load_pickle
plt.style.use('seaborn')


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


def save_model():
    return


def load_model():
    return


def build_model(timesteps, n_features, lr, gru_units, fc_units, output_shape):
    K.clear_session()
    model = Sequential()
    model.add(GRU(units=gru_units, input_shape=(timesteps, n_features,), return_sequences=True))
    model.add(GRU(units=gru_units, return_sequences=False))
    model.add(Dense(units=fc_units, activation='relu'))
    model.add(Dense(units=output_shape, activation='softmax'))
    model.compile(optimizer=Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


X_padded = pad_sequences(X, dtype=float, padding='post')
y_padded = pad_sequences(y, value=4, padding='post')
print(X_padded.shape)

n_unique_targets = np.unique(y_padded).shape[0]


# TODO: move to config
lr = 0.01
batch_size = 64
gru_units = 256
input_dim = 46  # represents a sequence length.
n_features = 256  # number of featues, vectors representing voice characteristics
output_shape = n_unique_targets
N = 1000


model = build_model(input_dim, n_features, lr=lr, gru_units=gru_units, fc_units=64, output_shape=input_dim)
model.summary()


model.fit(X_padded, y_padded, batch_size=batch_size, epochs=100)


plt.plot(np.mean(X_padded[1], axis=1))

plt.plot(y_padded[1])
