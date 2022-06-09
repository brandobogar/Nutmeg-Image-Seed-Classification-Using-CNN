from tensorflow.python.keras.saving.save import load_model
from dataset import data_uji, data_a, data_b, data_c
import numpy as np
import time

def model_evaluasi(model):
    model_load = load_model(model)

    test_loss, test_acc = model_load.evaluate(
        data_uji,
        verbose =0)
    test_acc = test_acc *100
    test_loss = test_loss *100

    print("Model yg di pakai :", model)
    #print('acuracy', test_acc)
    #print('loss', test_loss)
    print("accuracy = {:.1f}%".format(test_acc))
    print("loss = {:.1f}%".format(test_loss))
    print('--------------------------------------')

def model_summary(model):

    model_load = load_model(model)

    print('Model yg di pakai :', model)
    model_load.summary()
    #print('--------------------------')

def prediksi(model,data,pala):
    model_load = load_model(model) 
    prediksi = model_load.predict(data)
    prediksi = np.argmax(prediksi,axis=1)
    jumlah = len(prediksi)
    print('Model :', model)
    print('Prediksi data:', pala)
    print('Jumlah data :', jumlah)
    print('Hasil Prediksi:')
    print(prediksi)
    print('')

model = 'Model.h5'
eva= model_evaluasi(model=model)
summ = model_summary(model=model)

#prediksia = prediksi(model=model, data=data_a, pala="Pala A")
#prediksib = prediksi(model=model, data=data_b, pala="Pala B")
#prediksic = prediksi(model=model, data=data_c, pala="Pala C")

mod = model_summary(model=model)
#prediksia= prediksi(model='Model-4.h5',data=data_a,pala='Pala A')
#prediksib= prediksi(model='Model-4.h5',data=data_b,pala='Pala B')
#prediksic= prediksi(model='Model-4.h5',data=data_c,pala='Pala C')

#prediksia= prediksi(model='model-6.h5',data=data_a,pala='Pala A')
#prediksib= prediksi(model='model-6.h5',data=data_b,pala='Pala B')
#prediksic= prediksi(model='model-6.h5',data=data_c,pala='Pala C')

#model= model_summary(model='model-1.h5')
#model= model_summary(model='model-4.h5')
#model= model_summary(model='model-6.h5')

#model = model_evaluasi(model="Model-1.h5")
#model = model_evaluasi(model="Model-2.h5")
#model = model_evaluasi(model="Model-3.h5")
#model = model_evaluasi(model="Model-4.h5")
#model = model_evaluasi(model="Model-5.h5")
#model = model_evaluasi(model="Model-6.h5")
#model = model_evaluasi(model="Model-7.h5")




