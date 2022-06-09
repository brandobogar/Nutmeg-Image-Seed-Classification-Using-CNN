from cv2 import data
import tensorflow as tf 
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import Sequential, callbacks
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization
from tensorflow.python.keras.layers.core import Activation, Dropout
from tensorflow.python.keras.saving.save import load_model
from dataset import data_train,data_uji, data_validasi
import matplotlib.pyplot as plt
import os
import numpy as np

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
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    check_path = 'model\data\model.h5'
    checkpoint = ModelCheckpoint(
        check_path,
        monitor='accuracy',
        verbose=0,
        mode='max',
        save_best_only=True,
        save_weights_only=True,
        save_freq =1
    )

    history = model.fit(
        data_train,
        batch_size=32,
        validation_data=data_validasi,
        epochs=10,
        verbose=1,
        callbacks =[checkpoint])
    
    
    model = model.save('Model.h5')
    
    plt.figure()
    plt.subplot(211)
    plt.plot(history.history['accuracy'], label = 'accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.title("Grafik perbandingan Loss dan Akurasi")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim([0.1,1])
    plt.xlim([0,25])
    plt.legend(['Akurasi','Val_accuracy'], loc='lower right')

    plt.subplot(212)
    plt.plot(history.history['loss'], label = 'loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0.1,1])
    plt.xlim([0,25])
    plt.legend(['Loss','Val_loss'], loc='lower right')
    plt.tight_layout()
    plt.savefig('Model.png')

model = create_model()




