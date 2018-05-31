import numpy as np
import cv2
from .utils import *

# from processor import Processor
from .processor import Processor

class SegmentProcessor(Processor):
    def __init__(self, camera, background, stride, max_segments, max_rectangle=0.25, channel=None, hsv_delta=[100, 110, 16], name=None):#TODO: mess with this max_rectangle and see if that helps the big brect isues
        '''Pass stride = -1 to only process on request'''
        if(name is None):
            self.name_str = ''
        else:
            self.name_str = name
        super().__init__(camera, [

                                  self.name_str + 'bg-subtract',
                                  self.name_str + 'bg-filter-blur',
                                  self.name_str + 'bg-thresh',
                                  self.name_str + 'bg-erode',
                                  self.name_str + 'bg-open',
                                  self.name_str + 'distance',
                                  self.name_str + 'boxes',
                                  self.name_str + 'watershed'
                                  ], max(1, stride), name=name)
        self.rect_iter = range(0)
        self.background = background
        self.max_segments = max_segments
        self.max_rectangle = max_rectangle
        self.own_process = (stride != -1)
        self.channel = channel

        if channel is not None:
            # convert channel specification to an HSV value
            color = [0, 0, 0]
            color[channel] = 255
            hsv = cv2.cvtColor(np.uint8(color).reshape(-1,1,3), cv2.COLOR_BGR2HSV).reshape(3)
            # create an interval from that
            h_min = max(0, hsv[0] - hsv_delta[0])
            h_max = min(255, hsv[0] + hsv_delta[0])
            #swap them in case of roll-over
            #h_min, h_max = min(h_min, h_max), max(h_min,h_max)
            self.hsv_min = np.array([h_min, max(0,hsv[1] - hsv_delta[1]), hsv_delta[2]], np.uint8)
            self.hsv_max = np.array([h_max, 255, 255], np.uint8)

            print('range of color hsv', self.hsv_min, hsv, self.hsv_max)


    async def process_frame(self, frame, frame_ind):
        '''we only process on request'''
        if self.own_process:
            self._process_frame(frame, frame_ind)
            return
        return

    def _process_frame(self, frame, frame_ind):
        bg = self._filter_background(frame)
        dist_transform = self._filter_distance(bg)
        self.rect_iter = self._filter_contours(dist_transform, frame.shape)
        return

    def segments(self, frame = None):
        if frame is not None:
            self._process_frame(frame, 0)
        yield from self.rect_iter


    def _filter_background(self, frame, name = ''):

        img = frame#.copy()
        gray = cv2.UMat(img)
        #print('frame is type {} and self.background is type {}'.format(frame, self.background))
        if(self.background is not None):
            gray = diff_blur(self.background, frame, False)
        if name.find('bg-subtract') != -1:
            return gray
        if self.channel is None or True:
            if len(img.shape) == 3 and img.shape[2] == 3:
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        else:
            gray = gray#cv2.inRange(gray, self.hsv_min, self.hsv_max)
        gray = cv2.blur(gray, (5,5))
        if name.find('bg-filter-blur') != -1:
            return gray
        ret, bg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #bg = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                cv2.THRESH_BINARY_INV,11,2)
        if np.mean(cv2.mean(bg)) > 255 // 2:
           bg = cv2.subtract(bg, 255)

        if name.find('bg-thresh') != -1:
            return bg
        # noise removal
        kernel = np.ones((4,4),np.uint8)
        bg = cv2.erode(bg, kernel, iterations = 1)
        if name.find('bg-erode') != -1:
            return bg
        bg = cv2.morphologyEx(bg,cv2.MORPH_OPEN,kernel, iterations = 1)
        if name.find('bg-open') != -1:
            return bg

        return bg

    def _filter_distance(self, frame):
        dist_transform = cv2.distanceTransform(frame, cv2.DIST_L2,5)
        dist_transform = cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)

        #create distance tranform contours
        #dist_transform = np.uint8(cv2.UMat.get(dist_transform))
        dist_transform = np.uint8(dist_transform)
        return dist_transform

    def _filter_ws_markers(self, frame):
        _, contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #create markers
        markers = np.zeros( frame.shape, dtype=np.uint8 )

        for i in range(len(contours)):
            #we draw onto our markers with fill to create the mask
            cv2.drawContours(markers, contours, i, (i + 1,), -1)
        #draw a tiny circle to indicate background hint
        cv2.circle(markers, (5,5), 3, (255,))
        return markers.astype(np.int32)

    def sort_key(self, c):
        '''Get the area of a bounding rectangle'''
        rect = cv2.boundingRect(c)
        return rect[2] * rect[3]

    def _filter_contours(self, frame, frame_shape, return_contour=False):
        _, contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key = self.sort_key, reverse=True)
        rects = [cv2.boundingRect(c) for c in contours]
        segments = 0

        for c,r in zip(contours, rects):
            #flip around our rectangle
            # exempt small or large rectangles
            if(r[2] * r[3] < 250 or \
                r[2] * r[3] / frame_shape[0] / frame_shape[1] > self.max_rectangle ):
                continue
            if not return_contour:
                yield r
            else:
                yield c
            segments += 1
            if(segments == self.max_segments):
                break


    def _watershed(self, frame, markers):
        ws_markers = cv2.watershed(frame, markers)
        segments = 0
        for i in range(1, np.max(ws_markers)):
            pixels = np.argwhere(ws_markers == i)
            rect = cv2.boundingRect(pixels)
            #flip around our rectangle
            rect = (rect[1], rect[0], rect[3], rect[2])
            # exempt small or large rectangles (> 25 % of screen)
            if(len(pixels) < 5 or rect[2] * rect[3] < 100 or \
                rect[2] * rect[3] / frame.shape[0] / frame.shape[1] > self.max_rectangle ):
                continue
            yield rect
            segments += 1
            if(segments == self.max_segments):
                break

    def polygon(self, frame, rect = None):
        '''
            rect: an optional view which will limit the frame
        '''
        bg = self._filter_background(frame)
        dist_transform = self._filter_distance(bg)
        # filter herre
        if rect is not None:
            dist_transform = rect_view(dist_transform, rect)
            frame = rect_view(frame, rect)

        markers = self._filter_ws_markers(dist_transform)
        ws_markers = cv2.watershed(frame, markers)

        #sort based on size
        pixels = [np.flip(np.argwhere(ws_markers == i), axis=1) for i in range(1, np.max(ws_markers))]
        def key(x):
            r = cv2.boundingRect(x)
            return r[2] * r[3]
        pixels.sort(key = key, reverse=True)
        # add a polygon of the whole rect first
        result = []
        segments = 0
        for p in pixels:
            # exempt small rectangles
            rect = cv2.boundingRect(p)
            if(len(p) < 5 or rect[2] * rect[3] < 20 ):
                continue
            # once we find one, use it
            hull = cv2.convexHull(p)
            result.append((hull, rect))

            segments += 1
            if segments > self.max_segments:
                break

        #This code doesn't seem to fill the rectangle and I cannot figure out why
        result.append(np.array([
                     [0,0],
                    [0,frame.shape[0]],
                    [frame.shape[1], frame.shape[0]],
                    [frame.shape[1], 0],
                    ], np.int32).reshape(-1, 1, 2))
        return result

    async def decorate_frame(self, frame, name):
        if self.name_str not in name:
            return frame
        bg = self._filter_background(frame, name)


        dist_transform = self._filter_distance(bg)
        if 'distance' in name:
            return dist_transform

        if 'boxes' in name:
            for rect in self._filter_contours(dist_transform, frame.shape):
                draw_rectangle(frame, rect, (255, 255, 0), 1)
        if 'watershed' in name:
            markers = self._filter_ws_markers(dist_transform)
            ws_markers = cv2.watershed(frame, markers)
            frame[ws_markers == -1] = (255, 0, 0)
        return frame#cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
