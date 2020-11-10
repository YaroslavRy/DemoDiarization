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


def build_model(timesteps, lr, gru_units, fc_units, output_shape):
    K.clear_session()
    model = Sequential()
    model.add(GRU(units=gru_units, input_shape=(timesteps, 1,), return_sequences=False))
    model.add(Dense(units=fc_units, activation='relu'))
    model.add(Dense(units=output_shape, activation='softmax'))
    model.compile(optimizer=Adam(lr), loss='sparse_categorical_crossentropy')
    return model


# TODO: move to config
lr = 0.001
batch_size = 32
input_dim = 100  # represents a sequence length.
hidden_dim = 256
output_shape = 3
n_layers = 2
N = 1000

X = np.random.normal(size=(N, input_dim, 1))
y = np.random.randint(0, 3, size=(N, 1))

plt.plot(X.flatten())
plt.show()

model = build_model(input_dim, lr=lr, gru_units=32, fc_units=16, output_shape=output_shape)
model.summary()

model.fit(X, y, epochs=1)
