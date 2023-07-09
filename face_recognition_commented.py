# Face Recognition

import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # We load the cascade for the face.
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') # We load the cascade for the eyes.

#cascade is a series of filters that we will apply one after the to detect the face.
#we are going to the face detection on the global and main refertial.but, we are going to the eye detection on the referential of the face that will save us computational time.

#this function will use those cascades to detect the face and the eye.
def detect(gray, frame): # We create a function that takes as input the image in black and white (gray) and the original image (frame), and that will return the same image with the detector rectangles. 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # grey and white image, scale factor (by which image is scaled down), minimum no of neighbours
    # We apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
    #faces are tuple of 4 elements x,y which are coordinates of the upper left corner of the rectangle that will detect the face, width, height of the rectangle.
    
    #now we will iterate through different faces and for each of these faces we will draw the rectangle and inside these rectangles will detect some eyes.
    for (x, y, w, h) in faces: # For each detected face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # We paint a rectangle around the face.
        #frame where we want to make the rectangle, coordinates of upper left corner of the rectangle, coordinates of the lower right corner of the rectangle, 
        #color of rectangle and for that we need to use RGV code, thickness of the edges of the rectangle.
        
        #we are going to make the eyes rectangle inside the big rectangle.
        #now, we have zone of our interest which corresponds exactly to the zone inside the detector rectangle.
        roi_gray = gray[y:y+h, x:x+w] # We get the region of interest in the black and white image.
        roi_color = frame[y:y+h, x:x+w] # We get the region of interest in the colored image.
        
        #we are going to detect the eyes in the region of interest of gray omage becoz cascade is applied in the gray image.
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3) # We apply the detectMultiScale method to locate one or several eyes in the image.
        
        #we start a new for loop where we make a new rectangle around the eyes.
        for (ex, ey, ew, eh) in eyes: # For each detected eye:
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 2) # We paint a rectangle around the eyes, but inside the referential of the face.
    
    #we return original image with detection of both face and eyes.
    return frame # We return the image with the detector rectangles.

video_capture = cv2.VideoCapture(0) # We turn the webcam on.
#0 if comes from the internal webcam of this device.
#1 if webcam comes from the external device.

while True: # We repeat infinitely (until break):
    
    #read methods return two elements but we only need the second one i.e last frame of the webcam.
    _, frame = video_capture.read() # We get the last frame.
    
    #to get black and white version of our image.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # We do some colour transformations.
    canvas = detect(gray, frame) # We get the output of our detect function.
    cv2.imshow('Video', canvas) # We display the outputs.
    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
        break # We stop the loop.

video_capture.release() # We turn the webcam off.
cv2.destroyAllWindows() # We destroy all the windows inside which the images were displayed.



