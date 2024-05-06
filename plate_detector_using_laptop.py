import re
import os
import cv2
import imutils
import  pytesseract
import numpy as np
from PIL import ImageEnhance
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

DATA_DIR = './data'
#os.makedirs(DATA_DIR)
dataset_size = 10
cap = cv2.VideoCapture(0)
print('Collecting data')
done = False
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) == ord('q'):
        break

counter = 0
while counter < dataset_size:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(25)
    cv2.imwrite(os.path.join(DATA_DIR,'{}.jpg'.format(counter)), frame)
    counter += 1
cap.release()
cv2.destroyAllWindows()


plate_dict = {}
for count in range(10):
    img = cv2.imread(r"data"+"\\"+str(count)+".jpg")
    img = cv2.flip(img,1)
    
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
        image = Image.fromarray(cropped_img)
        image = ImageEnhance.Color(image).enhance(1.25)
        image = ImageEnhance.Sharpness(image).enhance(1.5)
        plate = np.array(image)
        text = pytesseract.image_to_string(plate)
        text = re.sub(r'\W+', '', text)
        if text in plate_dict:
            plate_dict[text]+=1
        else: plate_dict[text]=1
    except Exception as E:
        print(E)


for i in range(10): os.remove(r"data"+"\\"+str(i)+".jpg")
print(plate_dict)



    
