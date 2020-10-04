import numpy as np
import cv2
import face_recognition
import os
from datetime import datetime
import pandas as pd


path= 'attendancesystemimages'
images= []
classNames= []
mylist=os.listdir(path)
print(mylist)
for cls in mylist:
    currimg=cv2.imread(f'{path}/{cls}')
    images.append(currimg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

def findencodings(images):
    encodelist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

def markattendance(name):
    with open('attendance.csv','r+') as f:
        mydatalist=f.readline()
        namelist = []
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtstring=now.strftime('%H:%M:%S')
            datstring=now.strftime('%d %B, %Y')

            f.writelines(f'\n{name}, {datstring}, {dtstring}')



encodelistknown=findencodings(images)
print(len(encodelistknown))

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_current_frame = face_recognition.face_locations(imgS)
    enc_current_frame = face_recognition.face_encodings(imgS,faces_current_frame)

    for encodeface,location in zip(enc_current_frame,faces_current_frame):
        matches=face_recognition.compare_faces(encodelistknown,encodeface)
        facedis=face_recognition.face_distance(encodelistknown,encodeface)
        print(facedis)
        matchindex = np.argmin(facedis)
        if matches[matchindex]:
            name=classNames[matchindex]
            print(name)
            y1,x2,y2,x1=location
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0, 255, 120),2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX,1, (0,0,0),2 )
            markattendance(name)
    cv2.imshow('webcam',img)
    cv2.waitKey(0)