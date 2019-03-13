import numpy as np
import asyncio, cv2
from .utils import *

from .processor import Processor

class LineDetectionProcessor(Processor):
    ''' Detects drawn lines on an image (NB: works with a red marker or paper strip)
        This will not return knowledge of connections between reactors (that logic should be in DetectionProcessor or TrackerProcessor, which this class should be controlled by)
    '''
    def __init__(self, camera, stride, background, obsLimit = 5):
        super().__init__(camera, ['image-segmented','lines-detected'],stride)
        self._lines = [] # initialize as an empty array - list of dicts that contain endpoints, slope, intercept
        # preprocess the background image to help with raster noise
        #cv2.bilateralFilter(background, 7, 150, 150) #switched to median b/c bilateral preserves edges which is not what we want
        self._background = cv2.blur(cv2.medianBlur(background, 5), (7,7))
        self._ready = True
        self._observationLimit = obsLimit # how many failed calculations/countdowns until we remove a line
        self._stagedLines = [] # lines that were detected in the previous call to process_frame.  if they are detected again, add them to the main line list

    @property
    def lines(self):
        return self._lines

    async def process_frame(self, frame, frame_ind):
        if(self._ready):
            #copy the frame into it so we don't have it processed by later methods
            asyncio.ensure_future(self.detect_adjust_lines(frame.copy()))
        return

    async def decorate_frame(self, frame, name):
        smaller_frame = frame
        if name != 'image-segmented' or name != 'lines-detected':
            return  smaller_frame#cv2.cvtColor(smaller_frame, cv2.COLOR_BGR2GRAY)

        if name == 'image-segmented':
            bg = self.threshold_background(frame)
            return  bg#cv2.cvtColor(smaller_frame, cv2.COLOR_BGR2GRAY)

        # if name == 'lines-detected':
        #     print('Adding the points')
        # add purple points for each endpoint detected
        for i in range(0,len(self._lines)):
            cv2.circle(frame, (self._lines[i][0][0], self._lines[i][0][1]), (255,0,255),-1)
            cv2.circle(frame, (self._lines[i][1][0], self._lines[i][1][1]), (255,0,255),-1)

        return  frame#cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    '''
    Use _detect_lines to get currently found lines, and compare to the previously found ones.  Adjust/add/remove from _lines property
    Should return nothing, but updates the lines property
    '''
    async def detect_adjust_lines(self,frame):
        self._ready = False
        detected_lines = self._detect_lines(frame)#list of tuples of pair-tuples (the line endpoint coords)
        # need a way to remove previous lines that were not found.
        currentLines = self._lines
        # empty out self._lines.
        self._lines = []
        leftoverLines = [] # lines that did not match previously staged or currently held lines.  this becomes the next staged lines
        for i in range(0,len(currentLines)):
            # add/adjust a value that indicates if the current line was detected in this latest frame
            currentLines[i]['detected'] = False

        for i in range(0,len(detected_lines)):
            detectedEndpoints = detected_lines[i]
            (detectedSlope, detectedIntercept ) = line_from_endpoints(detectedEndpoints)
            # iterate through existing lines - we should store their slope and intercept
            mainLineUpdated = False

            # if self._lines is empty, it should just add the new line
            for j in range(0,len(currentLines)):
                existingLine = currentLines[j]

                currentSlope,currentIntercept = line_from_endpoints(existingLine['endpoints'])
                if(abs(percent_diff(currentSlope,detectedSlope)) < 0.15 and abs(percent_diff(currentIntercept,detectedIntercept)) < .15):
                    # easy way out - take the longer of the two lines. future implementations should check if they should be overlapped and take the longest line combination from them
                    if (distance_pts(detectedEndpoints) > distance_pts(existingLine['endpoints'])):
                        # replace with the new line
                        currentLines[j] = {'endpoints':detectedEndpoints, 'slope':detectedSlope, 'intercept':detectedIntercept, 'detected':True, 'observed':self._observationLimit}
                    else:
                        # update an existing line, so we know it was actually found
                        currentLines[j]['detected'] = True
                        # restart its countdown
                        currentLines[j]['observed'] = self._observationLimit
                    #however, at this stage, with whatever happens, we should stop iterating over existing lines
                    mainLineUpdated = True
                    break

            if not mainLineUpdated:
                # check if this line matches a staged line. if yes, increment its matching staged line's counter.  if not, add it as a new staged line
                stagedLineUpdated = False
                for j in range(0, len(self._stagedLines)):
                    stagedLine = self._stagedLines[j]
                    stagedSlope = stagedLine['slope']
                    stagedIntercept = stagedLine['intercept']
                    if (abs(percent_diff(stagedSlope,detectedSlope)) < 0.15 and abs(percent_diff(stagedIntercept, detectedIntercept)) < 0.15):
                        stagedLineUpdated = True
                        if(distance_pts(detectedEndpoints) > distance_pts(stagedLine['endpoints'])):
                            self._stagedLines[j] = {'endpoints':detectedEndpoints, 'slope':detectedSlope, 'intercept':detectedIntercept, 'detected':True, 'observed':self._observationLimit}
                        else:
                            self._stagedLines[j]['detected'] = True # staged does not decrement the observation limit, so no need to reset it
                        # stop iterating over staged lines
                        break
                if not stagedLineUpdated:
                    leftoverLines.append({'endpoints':detectedEndpoints, 'slope':detectedSlope, 'intercept':detectedIntercept, 'detected':False, 'observed':self._observationLimit})

        # one last passthrough, only adding lines that were re-detected or new
        for i in range(0,len(currentLines)):
            lineDict = currentLines[i]
            if (lineDict['detected']):
                self._lines.append(lineDict)

            else:
                # the line was not observed.  decrement the countdown, and only add it into our tracked lines if the observed value is greater than 0
                lineDict['observed'] -= 1
                if (lineDict['observed'] > 0):
                    self._lines.append(lineDict)

        # add any staged lines to the main list if they were detected again. set the leftover lines to be the new staged lines
        for i in range(0,len(self._stagedLines)):
            lineDict = self._stagedLines[i]
            if (lineDict['detected']):
                self._lines.append(lineDict)
        self._stagedLines = leftoverLines
        self._ready = True



    ''' Detect lines using filtered contour detection on the output of threshold_background
    '''
    def _detect_lines(self,frame):
        mask = self.threshold_background(frame)
        lines = []
        # detect contours on this mask
        _, contours, _ = cv2.findContours(mask, 1,cv2.CHAIN_APPROX_SIMPLE)
        if (contours is not None):
            for i in range(0, len(contours)):
                rect = cv2.minAreaRect(contours[i])
                # rect is a Box2D struct containing (x,y) as the center of the box, (w,h) as the width and height, and theta as the rotation
                area = rect[1][0]*rect[1][1]
                minDim = min(rect[1]) # the thickness of the line - we want to throw out rectangles that are too big
                maxDim = max(rect[1]) #corresponds to length - we want to throw out any noisy points that are too small
                if (rect[1][0] != 0 and rect[1][1] != 0):
                    aspectRatio = float(min(rect[1]))/max(rect[1])
                else:
                    aspectRatio = 100

                # we want a thin object, so a small aspect ratio.
                aspect_ratio_thresh = 0.3
                area_thresh_upper = 0.02 * frame.shape[0] * frame.shape[1]
                area_thresh_lower = 0.0002 * frame.shape[0] * frame.shape[1]
                width_thresh = 0.07 * frame.shape[0]
                length_thresh_lower = 0.05 * frame.shape[0]
                length_thresh_upper = 0.4 * frame.shape[0]
                if (aspectRatio < aspect_ratio_thresh and val_in_range(area, area_thresh_lower, area_thresh_upper) and minDim < width_thresh and val_in_range(maxDim, length_thresh_lower, length_thresh_upper)):
                    # only keep endpoints if it is the correct shape
                    endpoints = rect_to_endpoints(rect)
                    lines.append(endpoints)

        return lines


    def threshold_background(self,frame):
        sum_diff = diff_blur(self._background, frame)
        # threshold this value- play with thresh_val in prod
        thresh_val = 45
        _,mask = cv2.threshold(sum_diff, thresh_val, 255, cv2.THRESH_BINARY)
        # apply a sharpening filter
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        mask = cv2.filter2D(mask,-1,kernel)
        return mask

    def isLineSimilar(detectedEndpoints,currentEndpoints):
        detectedSlope,detectedIntercept = line_from_endpoints(detectedEndpoints)
        currentSlope,currentIntercept = line_from_endpoints(currentEndpoints)
        if(math.abs(percent_diff(currentSlope,detectedSlope)) < 0.05 and math.abs(percent_diff(currentIntercept,detectedIntercept)) < .05):
            return True
        else:
            return False
