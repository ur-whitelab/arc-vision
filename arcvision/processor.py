import asyncio
import random
import cv2
import numpy as np


class ExampleProcessor:
    '''Class that detects multiple objects given a set of labels and images'''
    def __init__(self, width=500, height=500, stride=1):
        self.corners = []
        self.corners.append((random.randrange(0, width / 4), random.randrange(0, height / 4)))
        self.corners.append((random.randrange(3 * width / 4, width), random.randrange(3 * height / 4, height)))
        self.color = tuple([random.randrange(0,255) for i in range(3)])
        self.stride = stride

    async def process_frame(self, frame):
        '''Perform update on frame, carrying out algorithm'''
        #simulate working hard
        return frame

    async def decorate_frame(self, frame):
        '''Draw visuals onto the given frame, without carrying-out update'''
        cv2.rectangle(frame, *self.corners, self.color, 2)
        return frame

class PreprocessProcessor:
    '''Substracts and computes background'''
    def __init__(self, stride=1, background=None):
        self.stride = stride
        self.background = background

    async def process_frame(self, frame):
        '''Perform update on frame, carrying out algorithm'''
        if self.background is not None:
            cv2.addWeighted(frame, 1.0, self.background, -1.0)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    async def decorate_frame(self, frame):
        return await self.process_frame(frame)

    def stop(self):
        pass


class DetectionProcessor:
    def __init__(self, query_images=[], labels=None, stride=1, threshold=0.8, template_size=(32,32), min_match=3):
        #load template images
        if labels is None:
            labels = ['Key {}'.format(i) for i in range(len(query_images))]
        #read, convert to gray, and resize template images
        self.templates = [{'name': k, 'img': cv2.resize(cv2.imread(i, 0), template_size), 'path': i} for k,i in zip(labels, query_images)]

        self.stride = 1
        self._ready = True
        self.features = {}
        self.threshold = threshold
        self.min_match = min_match

        # Initiate ORB detector
        self.orb = cv2.ORB_create()

        #get descriptors for templates
        for t in self.templates:
            t['orb'] = self.orb.detectAndCompute(t['img'], None)

        #set-up our flann detector
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        self.flann = cv2.FlannBasedMatcher(index_params, search_params)


    async def process_frame(self, frame):
        if(self._ready):
            #copy the frame into it so we don't have it processed by later methods
            asyncio.ensure_future(self._identify_features(frame.copy()))
        return frame

    async def decorate_frame(self, frame):
        for n in self.features:
            for f in self.features[n]:
                cv2.polylines(frame,f,True,255,3, cv2.LINE_AA)
                #get bottom of polygon
                y = max([p[1] for p in f])
                x = min([p[0] for p in f])
                cv2.putText(frame, n, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        return frame

    async def _identify_features(self, frame):
        '''This method tries to run the calculation over multiple loops.
            The _ready is to in lieu of a callback on completion'''
        self._ready = False
        #make new features object
        features = {}
        for t in self.templates:
            features[t['name']] = []
        for t in self.templates:
            template = t['img']
            name = t['name']
            descriptors = t['orb']
            w, h = template.shape[::-1]


            # find the keypoints and descriptors with orb
            kp2, des2 = self.orb.detectAndCompute(frame,None)

            matches = self.flann.knnMatch(descriptors[1],des2,k=2)

            # store all the good matches as per Lowe's ratio test.
            good = []
            for m,n in matches:
                if m.distance < self.threshold * n.distance:
                    good.append(m)
            if len(good) > self.min_match:
                src_pts = np.float32([ descriptors[0][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()

                h,w = img1.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,M)

            #cede control
            await asyncio.sleep(0)
        self._ready = True
        #now swap with our features
        self.features = features
