import cv2
import pytesseract
import numpy as np
from PIL import ImageGrab
import time

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files (x86)\Tesseract-OCR\\tesseract.exe'

img2 = cv2.imread('dtu.PNG')
img1=cv2.resize(img2,(512,512))
img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

#print(pytesseract.image_to_string(img))
# print(pytesseract.image_to_boxes(img))

hImg, wImg,_ = img.shape
# boxes = pytesseract.image_to_boxes(img)
# for b in boxes.splitlines():
#     print(b)
#     b = b.split(' ')
#     print(b)
#     x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
#     cv2.rectangle(img, (x,hImg- y), (w,hImg- h), (50, 50, 255), 2)
#     cv2.putText(img,b[0],(x,hImg- y+25),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,255),2)

boxes = pytesseract.image_to_data(img)
for a, b in enumerate(boxes.splitlines()):
    print(b)
    if a != 0:
        b = b.split()
        if len(b) == 12:
            x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
            cv2.putText(img, b[11], (x, y-2), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)
            cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 255), 2)

cv2.imshow('img', img)
cv2.waitKey(0)