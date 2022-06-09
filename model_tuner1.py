import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import Sequential
from tensorflow.python.keras import activations
from dataset import data_train, data_validasi, data_uji

def create_model():
    model = Sequential([
        Conv2D(64,kernel_size=(3,3),activation='relu'),
        MaxPooling2D(),
        Conv2D(64,kernel_size=(3,3),activation='relu'),
        MaxPooling2D(),
        Conv2D(64,kernel_size=(3,3),activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(3)
        ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    model.fit(data_train,epochs=10, validation_data=data_validasi)
    model_eval = model.evaluate(data_uji)

def model_builder(hp):
    '''
    Args:
    hp-keras tuner object
    '''
    model = keras.Sequential()
    model.add(keras.layers.Flattern(input_shape=(28,28)))
    hp_units = hp.Int('units', min_value=32, max_value=512, steps=32)
    model.add(keras.layers.Dense(units=hp_units, activation='relu', name='dense_1'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(10, activation='relu'))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
    return model

tuner = kt.Hyperband(
    model_builder, objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='dir',
    project_name='khyperband'
    )

tuner.search_space_summary()

    

#def build_model(hp):
    #model= keras.Sequential()
    #model.add(keras.layers.Dense(
        #hp.Choice('units',[8,16,32]),
        #activation='relu'
    #))
    #model.add(keras.layer.Dense(1, activation='relu'))
    #model.compile(loss='mse')
    #return model

#tuner = kkt.RandomSearch(
    #build_model,
    #objective='val_loss',
    #max_trials=5
#)

#tuner.search(data_train, epochs=5, validation_data=data_validasi)
#best_model = tuner.get_best_models()[0]