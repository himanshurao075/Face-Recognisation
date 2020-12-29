#Make sure you have already installed python and opencv, numpy in your pc
import cv2
import numpy as np

#Change Below Path According to your pc (Past full path till haarcascade_frantalface_defailt.xml  file

face_Classifer=cv2.CascadeClassifier('C:/Users/himanshrav/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')



def face_extracter(img):
    #converting color image into gray image
    grayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #This face_calssfier is extract  only face image through full image
    faces=face_Classifer.detectMultiScale(grayImage,1.1,1,minSize=(30,30))

    if faces is ():
        None
    for (x,y,w,h) in faces:
        #here x,y,w,h is dimenssion of extracted facial image
        cropped_faces=img[y:y+h,x:x+w]
        return cropped_faces



cam=cv2.VideoCapture(0)
#here count is variable to count frames
count=0


while True:
    ret, frame=cam.read()
    if face_extracter(frame) is not None:
        count+=1
        face=cv2.resize(face_extracter(frame),(200,200))
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        # fileNamePath  is the path of that folder where you want to save your face sample ,modify it according to your choice
        fileNamePath='C:/Users/himanshrav/Desktop/Face DataSets/FaceImage'+str(count)+'.jpg'
        cv2.imwrite(fileNamePath,face)

        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
    else:
        print("Face Not Found")
        pass
        # coutn=1000 denotes the number of face sample max sample increase the accuracy but increase execution time
    if cv2.waitKey(1)==13 or count==1000:
        break
#always release hardware before terminate execution
cam.release()
cv2.destroyAllWindows()
