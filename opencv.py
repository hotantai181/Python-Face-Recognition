import cv2
import face_recognition
import os
from datetime import datetime
import numpy as np

# load anh tu kho nhan dang

path ="pic"
images =[]
className =[]
myList = os.listdir(path)
print(myList)
for cl in myList:
    print(cl)
    curImg = cv2.imread(f"{path}/{cl}") #pic
    images.append(curImg)  #them vao bien images
    className.append(os.path.splitext(cl)[0])
    # splitext tach chuoi thanh 2 phan phan truoc duoi mo rong va phan duoi mo rong
print(len(images))
print(className)
#step encoding
def Mahoa(images):
    encodeList =[]
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # chuyen BGR sang RGB
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnow = Mahoa(images)
print("Ma hoa thanh cong")
print(len(encodeListKnow))

def docghi(name):
    with open("docghi.csv","r+") as f:
        myDatalist = f.readlines()
        nameList = []
        for line in myDatalist:
            entry = line.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtstring = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name},{dtstring}")

#Khoi dong webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    framS = cv2.resize(frame,(0,0),None,fx=0.5,fy=0.5)
    framS = cv2.cvtColor(framS, cv2.COLOR_BGR2RGB)  # chuyen BGR sang RGB
    #Xac dinh vi tri khuon mat tren cam

    facecurFrame = face_recognition.face_locations(framS) # Lay tung khuon mat va vi tri khuon mat
    encodecurFrame = face_recognition.face_encodings(framS)

    for encodeFace, faceLoc in zip(encodecurFrame,facecurFrame): #lay tung khuon mat theo vi tri hien tai
        matches = face_recognition.compare_faces(encodeListKnow,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis) # day ve gia tri index nho nhat
        print(matchIndex)

        if faceDis[matchIndex] <0.50:
            name = className[matchIndex].upper()
            docghi(name)
        else:
            name="Unknow"
        #print  ten tren anh
        y1 , x2 ,y2 ,x1 = faceLoc
        y1 , x2 ,y2 ,x1 = y1*2,x2*2,y2*2,x1*2
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,name,(x2,y2),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


    cv2.imshow('Nhan dang ',frame)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()