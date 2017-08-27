import os

import librosa.core as libcore
import numpy as np
from scipy.ndimage.interpolation import zoom

from keras.layers import Input, Dense, LSTM, GRU
from keras.layers.core import RepeatVector
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.callbacks import ModelCheckpoint

SAMPLING_RATE = 44100
N_FFT = 1024
SEQ_TIME = 5
NOISE_SCALE = 1.0e-5

CHUNK_SIZE = SEQ_TIME * SAMPLING_RATE
HOP_LENGTH = N_FFT // 4
STFT_FPS = SAMPLING_RATE // HOP_LENGTH

SEQ_LEN = (SEQ_TIME * STFT_FPS + 2) // 2
INPUT_DIM = (N_FFT // 4) // 2

WAV_FILE = '../data/raw/first.flac'
WEIGHT_FILE_PATTERN = '../output/first.gru-{epoch:d}-{val_loss:.3f}.hdf'

TS_SIZE = 1000
T_EPOCHS = 400
SAVE_AFTER = 10

def walk(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            yield os.path.join(root, name), name


def sample_chunk(samples, power, position=None, size=CHUNK_SIZE):
    if position is None:
        position = int(np.random.uniform(high=len(samples) - size))

    if NOISE_SCALE > 0:
        noise = np.random.normal(loc=0, scale=NOISE_SCALE, size=size)
        chunk = (samples[position: position + size] + noise) / power
    else:
        chunk = samples[position: position + size] / power

    spectrum = np.log(np.abs(libcore.stft(chunk, n_fft=N_FFT)) + 1)[0: N_FFT // 4] / 10.0
    spectrum = zoom(spectrum, zoom=0.5)

    return spectrum


def train():

    print('Loading dataset: {} ...'.format(WAV_FILE))

    samples, _ = libcore.load(WAV_FILE, sr=SAMPLING_RATE)
    power = np.mean(samples ** 2) * 0.5

    print('Sampling training set nb_samples={}, size=({},{}) ...'.format(TS_SIZE, SEQ_LEN, INPUT_DIM))
    training_set = np.array([sample_chunk(samples, power).T for _ in range(TS_SIZE)])

    print('Constructing autoencoder ...')

    inputs = Input(shape=(SEQ_LEN, INPUT_DIM))
    enc_1 = GRU(128)(inputs)
    features = RepeatVector(SEQ_LEN)(enc_1)
    dec_0 = GRU(128, return_sequences=True)(features)
    dec_1 = GRU(INPUT_DIM, return_sequences=True)(dec_0)
    autoencoder = Model(inputs, dec_1)

    autoencoder.summary()
    autoencoder.compile(optimizer='rmsprop', loss='mse')

    model_cb = ModelCheckpoint(WEIGHT_FILE_PATTERN, monitor='val_loss', verbose=0,
                               save_best_only=False, save_weights_only=False, mode='auto', period=SAVE_AFTER)

    print('Training autoencoder for {} epochs. Save each {}th epoch ...'.format(T_EPOCHS, SAVE_AFTER))

    history = autoencoder.fit(training_set, training_set, nb_epoch=T_EPOCHS, validation_split=0.1, callbacks=[model_cb])


if __name__ == '__main__':
    train()
