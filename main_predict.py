# import the necessary packages
from imutils import face_utils
import dlib
import cv2
from scipy.spatial import distance as dist
import pickle
import time
import time

classifier = open('classifier.pkl',"rb")
classifier = pickle.load(classifier)

classifier_dict= open('class_dict.pkl',"rb")
classifier_dict = pickle.load(classifier_dict)

def eqlidian_distance(mid,shapes):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    list_dist = list()
    for shape in shapes:
        A = dist.euclidean(mid, shape)
        #print(A)
        list_dist.append(A)
    try:
         prd = classifier.predict([list_dist])
         cname = classifier_dict[prd[0]]
         print(cname)
         return cname
    except Exception as e:
        print(e)
        return '...'
    #B = dist.euclidean(eye[2], eye[4])
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path)

cap = cv2.VideoCapture(0)

out = cv2.VideoWriter('vid.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (530,460))
 
while True:
    # load the input image and convert it to grayscale
    _, image = cap.read()
    image = cv2.flip(image,1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # detect faces in the grayscale image
    rects = detector(gray, 0)
    text = '^^_^^'
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        mid = shape[30]
        try:
            text = eqlidian_distance(mid, shape)
        except:
            print('err')
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        cv2.circle(image, tuple(mid), 3, (255, 255, 0), -1)
    cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 255), 2)
    
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    out.write(image)
    cv2.imwrite('./caps/'+str(time.time())+'.jpg', image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        #out.release()
        break

cv2.destroyAllWindows()

cap.release()
out.release()