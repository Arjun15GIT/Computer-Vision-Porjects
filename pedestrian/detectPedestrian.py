# People Detection using HOG and Non-Maximum Suppression
import cv2
from imutils.object_detection import non_max_suppression
from imutils import resize
import numpy as np

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

img = cv2.imread('pedestrian/360_F_224656606_5iJMRRdxrfsfbMQQlMeV7fOVjnG45Db0.webp')
img = resize(img,height=500)

rects,weights = hog.detectMultiScale(img,winStride=(4,4),padding=(8,8),scale=1.05)

copy = img.copy()
for x,y,w,h in rects:
    cv2.rectangle(copy,(x,y),(x+w,y+h),(0,0,255),2)

cv2.imshow('before suppression',copy)
cv2.waitKey(0)

r = np.array([[x,y,x+w,y+h] for x,y,w,h in rects])
pick = non_max_suppression(r,probs=None,overlapThresh=0.65)    

for xa,ya,xb,yb in pick:
    cv2.rectangle(img,(xa,ya),(xb,yb),(0,255,0),2)

cv2.imshow('after suppression',img)
cv2.waitKey(0)

cv2.destroyAllWindows()