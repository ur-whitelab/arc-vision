import asyncio
import cv2
import numpy as np
from imutils import contours
from skimage import measure
import imutils

class Detector:
    '''Class that detects multiple objects given a set of labels and images'''
    def __init__(self, camera, query_images=[], labels=None):
        if labels is None:
            labels = ['Key {}'.format(i) for i in range(len(query_images))]
        self.keys = [{'name': k, 'img': cv2.imread(i, 0), 'path': i} for k,i in zip(labels, query_images)]
        self.camera = camera

    def get_snapshot(self,camera,file_location):
        if camera.cap.isOpened():
            ret,frame=camera.cap.read()
            camera.save_frame(frame,file_location)    
        else:
            print('get_snapshot failed: camera is closed')  

    def frame_update(self, frame):
        '''Perform update on frame, carrying out algorithm'''
        newframe=self.get_snapshot(self.camera,file_location='temp/subject.png')
        self.centers,self.circles = identify_features(background='temp/background.png', subject='temp/subject.png')
        print(self.centers)
        #asyncio.sleep(5)

    def frame_decorate(self, frame):
        '''Draw visuals onto the given frame, without carrying-out update'''
        # print('FRAME_DECORATE',self.circles)
        pass

    def attach(self, camera, stride=1):
        camera.add_frame_fxn(self.frame_update, stride)
        camera.add_frame_fxn(self.frame_decorate, stride)




def identify_features(background='background1.png',subject='subject1.png'):
    img1=cv2.imread(background)
    img2=cv2.imread(subject)
    imgdiff=cv2.subtract(img1,img2)

    image = imgdiff
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
 
    labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")
    for label in np.unique(labels):
        # ignore background label
        if label == 0:
            continue
 
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
 
        if numPixels > 300:
            mask = cv2.add(mask, labelMask)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    if len(cnts)>0:
        cnts = contours.sort_contours(cnts)[0]
 
    # loop over the contours
    centers = []
    circles = []
    for (i, c) in enumerate(cnts):
        # draw the bright spot on the image
        #(x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        centers.append((cX,cY))
        circles.append(((cX, cY), radius))
        cv2.circle(image, (int(cX), int(cY)), int(radius),(0, 0, 255), 3)

    return (centers,circles)


def send_data():
    pass