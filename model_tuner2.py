import keras_tuner as kt
from keras_tuner import tuners
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import Sequential
from tensorflow.python.keras import activations
from dataset import data_train, data_validasi, data_uji

def build_model(hp):
    model=keras.Sequential()
    model.add(keras.layers.Conv2D(16,3,activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Conv2D(filters=hp.Choice(
        'num_filters',
        values=[32,64],
        default=64), kernel_size=3, activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(
        hp.Choice('units',[8,16,32]),
        activation='relu'
    ))
    model.add(keras.layers.Dense(1,activation='relu'))
    model.compile(loss='mse')
    return

tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5
)

tuner.search(data_train, epochs=5, validation_data=data_validasi)
best_model= tuner.get_best_models()[0]