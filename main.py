import cv2
import face_recognition
import  numpy as np
from tkinter import *
imgElon = face_recognition.load_image_file("pic/elon_musk2.JPG")
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
small_fram = cv2.resize(imgElon,(0,0),fx=0.8,fy=0.8)


imgCheck = face_recognition.load_image_file("pic/check.jpg")
imgCheck = cv2.cvtColor(imgCheck,cv2.COLOR_BGR2RGB)

#Vị trí khuôn mặt
faceLoc = face_recognition.face_locations(small_fram)[0]
print(faceLoc) #y1,x2,y2,x1
#Mã hóa hình ảnh
encodeElon = face_recognition.face_encodings(small_fram)[0]
cv2.rectangle(small_fram,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)



faceCheck = face_recognition.face_locations(imgCheck)[0]
encodeCheck = face_recognition.face_encodings(imgCheck)[0]
cv2.rectangle(imgCheck,(faceCheck[3],faceCheck[0]),(faceCheck[1],faceCheck[2]),(255,0,255),2)
result = face_recognition.compare_faces([encodeElon],encodeCheck)

# Sai số giữa các ảnh
faceDis = face_recognition.face_distance([encodeElon],encodeCheck)
print(result,faceDis)
cv2.imshow("Elon",small_fram)
cv2.imshow("Check",imgCheck)
cv2.waitKey()
window = Tk()
window.geometry('600x800')
lbl = Label(window, img = imgElon)
lbl.place(x=0,y=0)

window.mainloop()
