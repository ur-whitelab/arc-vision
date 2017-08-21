import asyncio
import time
import cv2
import numpy as np

class Calibrate:

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    '''initial position calibration'''
    def __init__(self):
        
        print('calibinit')

        self.objp = np.zeros((6*7,3), np.float32)
        self.objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
        self.objpoints = [] 
        self.imgpoints = [] 




    def calibrate_image(self,fname):
        print('calibup')
        img = fname #cv2.imread(fname)
        print(img)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,7), flags=cv2.CALIB_CB_ADAPTIVE_THRESH)
        print(corners)


        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7,7), corners2,ret)
            cv2.imshow('img',img)
            cv2.waitKey(50)

        else:
            print('CAPTURE FAILED')
