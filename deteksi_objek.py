import cv2 as cv
import numpy as np
import os

img1= cv.imread('benda_lain.jpg', cv.IMREAD_UNCHANGED)

img=cv.resize(img1,(500,500), interpolation=cv.INTER_AREA)
contrast = cv.convertScaleAbs(img, beta=31.0, alpha=1.1)

gray = cv.cvtColor(contrast, cv.COLOR_BGR2GRAY)

mask=cv.threshold(gray, 127,255, cv.THRESH_BINARY)[1]

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
#cnt1 = cv.drawContours(img, cnts, -1, (0,255,0), 3)

cv.imshow("Deteksi Objek", outline)
res = cv.putText(img, "Jumlah objek: %i " % len(cnts), (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv.LINE_AA)
#cv.imwrite('Kepadatan Rendah.jpg', res)
cv.imshow("Hasil Deteksi", res)

cv.waitKey(0)
cv.destroyAllWindows()