from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Input
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot as plt


def save_model():
    return


def laod_model():
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
