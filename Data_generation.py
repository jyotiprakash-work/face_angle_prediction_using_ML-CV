#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary packages
from imutils import face_utils
import dlib
import cv2
from scipy.spatial import distance as dist
import pickle
import time


# In[2]:


''' 
    this method is calculating equlidian distance beetween the face land marks. Here lavel and class name 
    will be changed for different face angles.
'''
def eqlidian_distance(mid,shapes):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    list_dist = list()
    for shape in shapes:
        A = dist.euclidean(mid, shape)
        print(A)
        list_dist.append(A)
    dict_data = {}
    dict_data['dist_data'] = list_dist
    dict_data['class'] = 4# change to 0 for right, 1 for left,2 for up and 3 for down 
    dict_data['class_name'] = 'front'# lable will be changed based on class index
    with open('data/front'+str(time.time())+'.pkl', 'wb') as f:
        pickle.dump(dict_data, f)


# In[3]:


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path)


# In[ ]:


cap = cv2.VideoCapture(0)
 
while True:
    # load the input image and convert it to grayscale
    _, image = cap.read()
    image = cv2.flip(image,1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # detect faces in the grayscale image
    rects = detector(gray, 0)
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    
        # loop over the (x, y)-coordinates for the facial landmarks
        # mid is 30 index 
        mid = shape[30]
        eqlidian_distance(mid, shape)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        cv2.circle(image, tuple(mid), 3, (255, 255, 0), -1)
    
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

