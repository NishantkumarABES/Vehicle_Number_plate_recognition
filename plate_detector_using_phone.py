import re
import os
import cv2
import imutils
import  pytesseract
import requests
import numpy as np
from PIL import ImageEnhance
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt



DATA_DIR = './data'
dataset_size = 10
url = "http://10.229.244.106:8080/shot.jpg"
print('Collecting data')
done = False
while True:
    img_resp = requests.get(url) 
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8) 
    img = cv2.imdecode(img_arr, -1) 
    frame = imutils.resize(img, width=1000, height=1800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) == ord('q'):
        break

counter = 0
while counter < dataset_size:
    img_resp = requests.get(url) 
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8) 
    img = cv2.imdecode(img_arr, -1) 
    frame = imutils.resize(img, width=1000, height=1800)
    cv2.imshow('frame', frame)
    cv2.waitKey(25)
    cv2.imwrite(os.path.join(DATA_DIR,'{}.jpg'.format(counter)), frame)
    counter += 1

cv2.destroyAllWindows()


plate_dict = {}
for count in range(10):
    img = cv2.imread(r"data"+"\\"+str(count)+".jpg")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray,11,17,17) #noise reduction
    edged = cv2.Canny(bfilter,30,200) #edge detection
    edged_rgb = cv2.cvtColor(edged,cv2.COLOR_BGR2RGB)

    keypoints = cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours,key = cv2.contourArea,reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour,10,True)
        if len(approx)==4:
            location = approx
            break
    mask = np.zeros(gray.shape,np.uint8)
    try:
        new_image = cv2.drawContours(mask,[location],0,255,-1)
        new_image = cv2.bitwise_and(img,img,mask=mask)
        (x,y) = np.where(mask==255)
        x1,y1 = np.min(x),np.min(y)
        x2,y2 = np.max(x),np.max(y)
        cropped_img = gray[x1:x2+1,y1:y2+1]
        plate = np.array(cropped_img)
        text = pytesseract.image_to_string(plate)
        text = re.sub(r'\W+', '', text)
        if text in plate_dict:
            plate_dict[text]+=1
        else: plate_dict[text]=1
    except Exception as E:
        print(E)


for i in range(10): os.remove(r"data"+"\\"+str(i)+".jpg")
print(plate_dict)



    
