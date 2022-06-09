import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import pathlib
import cv2 as cv
import os
from preprocessing_datatrain import preprocessing


data_train = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/data_training/data',
    image_size=(150,150),
    shuffle=True,
    validation_split=0.2,
    subset='training',
    seed=0,
    batch_size=32)

data_validasi = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/data_training/data',
    image_size=(150,150),
    shuffle=True,
    validation_split=0.2,
    subset='validation',
    seed=0,
    batch_size=32)

data_uji = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/data_uji/data',
    image_size=(224,224),
    shuffle=True,
    batch_size=32)

data_a = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/data_tes/pala/a',
    image_size=(224,224),
    shuffle=False,
    batch_size=32)

data_b = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/data_tes/pala/b',
    image_size=(224,224),
    shuffle=False,
    batch_size=32)

data_c = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/data_tes/pala/c',
    image_size=(224,224),
    shuffle=False,
    batch_size=32)






