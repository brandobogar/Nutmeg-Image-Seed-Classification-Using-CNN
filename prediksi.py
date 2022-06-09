import os
import glob
import tensorflow as tf
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Activation, Dense, Dropout, Flatten
from tensorflow.python.keras.layers.pooling import MaxPooling2D
import cv2 as cv
from dataset import data_uji,data_train
import numpy as np

def preprocessing(citra_tes):
    img1= cv.imread(citra_tes, cv.IMREAD_UNCHANGED)

    img=cv.resize(img1,(500,500), interpolation=cv.INTER_AREA)
    contrast = cv.convertScaleAbs(img, beta=31.0, alpha=1.1)

    gray = cv.cvtColor(contrast, cv.COLOR_BGR2GRAY)

    mask=cv.threshold(gray, 128,255, cv.THRESH_BINARY)[1]

    #mask = 255 - mask

    kernel = np.ones((10,10), np.uint8)

    mask=cv.morphologyEx(mask,cv.MORPH_OPEN,kernel)
    mask=cv.morphologyEx(mask,cv.MORPH_CLOSE,kernel)

    mask = cv.GaussianBlur(mask,(0,0), sigmaX=2,sigmaY=2,borderType=cv.BORDER_DEFAULT)

    mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)

    result = img.copy()
    result = cv.cvtColor(result, cv.COLOR_BGR2BGRA)
    result[:,:,3] = mask
    outline = cv.Canny(mask, 30, 150)

    (cnts, _) = cv.findContours(outline, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    sorted_contours= sorted(cnts, key=cv.contourArea, reverse= True)

    save_crop_path = 'dataset\data_tes\data\data'
    for file in os.scandir(save_crop_path):
        os.remove(file.path)

    os.chdir(save_crop_path)
    for (i,c) in enumerate (sorted_contours):
        x,y,w,h=cv.boundingRect(c)
        cropped_contour=img[y:y+h, x:x+w]
        image_name= "data_tes-" + str(i+1) + ".jpg"
        cv.imwrite(image_name,cropped_contour)
        readimage=cv.imread(image_name)
    root = r'D:\Document\Skripsi\1 Aplikasi v2'
    os.chdir(root)

def prediction(citra_tes,data_tes):
    img1= cv.imread(citra_tes, cv.IMREAD_UNCHANGED)

    img=cv.resize(img1,(500,500), interpolation=cv.INTER_AREA)
    contrast = cv.convertScaleAbs(img, beta=31.0, alpha=1.1)

    gray = cv.cvtColor(contrast, cv.COLOR_BGR2GRAY)

    mask=cv.threshold(gray, 128,255, cv.THRESH_BINARY)[1]

    #mask = 255 - mask

    kernel = np.ones((10,10), np.uint8)

    mask=cv.morphologyEx(mask,cv.MORPH_OPEN,kernel)
    mask=cv.morphologyEx(mask,cv.MORPH_CLOSE,kernel)

    mask = cv.GaussianBlur(mask,(0,0), sigmaX=2,sigmaY=2,borderType=cv.BORDER_DEFAULT)

    mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)

    result = img.copy()
    result = cv.cvtColor(result, cv.COLOR_BGR2BGRA)
    result[:,:,3] = mask
    outline = cv.Canny(mask, 30, 150)

    (cnts, _) = cv.findContours(outline, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    model_load = tf.keras.models.load_model('Model.h5')

    hasil =model_load.predict(data_tes)
    hasil = np.argmax(hasil, axis =1)

    for i in cnts:
        for i in range(len(hasil)):
            if hasil[i]==0:
                res = cv.drawContours(img, cnts[i], -1, (255, 0, ), 6)
            elif hasil[i]==1:
                res = cv.drawContours(img, cnts[i], -1, (0, 255, 0), 6)
            elif hasil[i]==2:
                res = cv.drawContours(img, cnts[i], -1, (0, 0, 255), 6)
    
    total_pala = len(hasil)
    pala_a=(hasil == 0).sum()
    pala_b=(hasil == 1).sum()
    pala_c=(hasil == 2).sum()
    res = cv.putText(img, "Jumlah pala: %i " % total_pala, (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv.LINE_AA)
    res = cv.putText(img, "Pala A: %i " % pala_a, (10,60), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv.LINE_AA)
    res = cv.putText(img, "Pala B: %i " % pala_b, (10,90), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv.LINE_AA)
    res = cv.putText(img, "Pala C: %i " % pala_c, (10,120), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv.LINE_AA)
    cv.imshow('Hasil_Akhir',res)
    root = r'D:\Document\Skripsi\1 Aplikasi v2'
    #root in this script is my root folder in my project
    os.chdir(root)
    cv.waitKey(0)
    cv.destroyAllWindows()

###INPUT IMAGE
citra = '30_biji.jpg'

###object detection
datates = preprocessing(citra_tes=citra)

#dataset   
data_tes = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/data_tes/data',
    image_size=(150,150),
    shuffle=False,
    batch_size=32)

###image prediksi
prediksi = prediction(citra_tes=citra, data_tes=data_tes)
    


