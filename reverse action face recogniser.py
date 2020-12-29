#make sure you  have already h install below library like subprocess, time, numpy opencv,os

import  subprocess as cmd
import time
import cv2
import numpy as np
from os import listdir
from os.path import join,isfile

# first train your model
TrainingData,Labels=[],[]
#datasetPath  is a path where your sample images are saved, modify these path according to your pc
datasetPath='C:/Users/himanshrav/Desktop/Face DataSets/'

#reading images from datasetPath and traing to model (model Name is :LBPHFaceRecogniser   (Local Binary Patterns Histogram))
onlyFiles=[f for f in listdir(datasetPath) if isfile(join(datasetPath,f))]
for i ,file in enumerate(onlyFiles):
    imagePath=datasetPath+onlyFiles[i]
    images=cv2.imread(imagePath,0)
    TrainingData.append(np.asarray(images,dtype='uint8'))
    Labels.append(i)
Labels=np.asarray(Labels,dtype='int32')
model=cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(TrainingData),np.asarray(Labels))
print("Model Training Complete!!")

#the above CaseCadeClassifier path is same as training file, Modify it according to your pc
face_Classifer=cv2.CascadeClassifier("C:/Users/himanshrav/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")



def face_detector(img,size=0.5):
    grayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_Classifer.detectMultiScale(grayImage,1.1,1)

    if faces is ():
        return img,[]

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi=img[y:y+h,x:x+w]
        roi2=cv2.resize(roi,(200,200))

    return img,roi2

cam=cv2.VideoCapture(0)
fnm=0

while True:

    ret, frame=cam.read()
    img, face=face_detector(frame)

    try:
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        result=model.predict(face)
        # This line is only for devloping purpose ignore it                       print("r=",result[1])

        if result[1]<500:
            confidence=int(100*(1-(result[1])/300))
            display_String=str(confidence)+'% Confidence it is User'
            confirmation_String='Face Not Match '

        cv2.putText(img,display_String,(100,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)

        if confidence > 85:
            cv2.putText(img,'Face Match',(225,460),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),1)
            cv2.imshow("Face Cropper",img)
            fnm=0



        else:
            cv2.putText(img,confirmation_String, (225, 450), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
            cv2.putText(img,'Screen Loack in '+str(fnm)+'th Frames', (200, 470), cv2.FONT_HERSHEY_COMPLEX,.6, (255, 255, 255), 1)
            cv2.imshow("Face Cropper", img)
            fnm+=1


            if fnm==50:
                #This rundll32... is command for lock PC
                cmd.call('rundll32.exe user32.dll,LockWorkStation')
                break

    except:
        cv2.putText(img, 'Face Not Found', (200, 460), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 1)
        cv2.imshow("Face Cropper", img)
        fnm=0
        pass

    if cv2.waitKey(1) == 13:
        break
#always release hardware  before terminate execution
cam.release()
cv2.destroyAllWindows()