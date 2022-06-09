import cv2 as cv
import os
import numpy as np

def preprocessing(awal,tujuan):
    #for file in os.scandir(tujuan):
        #os.remove(file.path)
    for filename in os.listdir(awal):
        img1 = cv.imread(os.path.join(awal,filename),cv.IMREAD_UNCHANGED)
        img=cv.resize(img1,(500,500), interpolation=cv.INTER_AREA)

        contrast = cv.convertScaleAbs(img, beta=31.0, alpha=1.1)

        gray = cv.cvtColor(contrast, cv.COLOR_BGR2GRAY)

        mask=cv.threshold(gray, 128,255, cv.THRESH_BINARY)[1]

        #mask = 255 - mask

        kernel = np.ones((20,20), np.uint8)

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

        os.chdir(tujuan)
        a=+1
        for (i,c) in enumerate (sorted_contours):
            x,y,w,h=cv.boundingRect(c)
            cropped=img[y:y+h, x:x+w]
            if cropped.shape[0] >=100 and cropped.shape[1]>=100:
                img_name = filename
                cv.imwrite(img_name,cropped)
    
        root = r'D:\Document\Skripsi\1 Aplikasi v2'
        os.chdir(root)

awal1 = 'img\pala_a'
awal2 = 'img\pala_b'
awal3 = 'img\pala_c'
awal4 = 'img\data_uji'
awal5 = 'img\data_tes'

tujuan1 = 'dataset\data_training\data\pala_a'
tujuan2 = 'dataset\data_training\data\pala_b'
tujuan3 = 'dataset\data_training\data\pala_c'
tujuan4 = 'dataset\data_uji\data\pala_a'
tujuan5 = 'dataset\data_uji\data\pala_b'
tujuan6 = 'dataset\data_uji\data\pala_c'
tujuan7 = 'dataset\data_tes\pala'

#pala_a = preprocessing(awal=awal1,tujuan=tujuan1)
#pala_b = preprocessing(awal=awal2,tujuan=tujuan2)
#pala_c = preprocessing(awal=awal3,tujuan=tujuan3)
#data_uji = preprocessing(awal = awal4, tujuan=tujuan4)
#data_tes = preprocessing(awal = awal5, tujuan=tujuan7)