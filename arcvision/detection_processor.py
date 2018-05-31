import numpy as np
import asyncio, cv2
from .utils import *

# from processor import Processor
from .processor import Processor

from .tracker_processor import TrackerProcessor

from .segment_processor import SegmentProcessor


class DetectionProcessor(Processor):
    '''Detects query images in frame. Uses async to spread out computation. Cannot handle replicas of an object in frame'''
    def __init__(self, camera, background, img_db, descriptor, stride=3,
                 threshold=0.8, template_size=256, min_match=6,
                 weights=[3, -1, -1, -10, 5], max_segments=10,
                 track=True):

        #we have a specific order required
        #set-up our tracker
        # give estimate of our stride
        if track:
            self.tracker = TrackerProcessor(camera, stride * 2 * len(img_db), background)
        else:
            self.tracker = None


        #then our segmenter
        self.segmenter = SegmentProcessor(camera, background, -1, max_segments)
        #then us
        super().__init__(camera, ['keypoints', 'identify'], stride)


        self._ready = True
        self.features = {}
        self.threshold = threshold#this is the percentage of distance similarity
        self.min_match = min_match
        self.weights = weights
        self.stretch_boxes=1.5
        self.track = track
        self.templates = img_db
        self.stride = stride

        # Initiate descriptors
        self.desc = descriptor
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params,search_params)#cv2.BFMatcher()

        #create color gradient
        N = len(img_db)
        for i,t in enumerate(self.templates):
            rgba = [int(x * 255) for x in np.random.random(size=4)]
            t.color = rgba[:-1]
            if t.keypoints is None:
                t.keypoints = self.desc.detect(t.img)
            t.keypoints, t.features = self.desc.compute(t.img, t.keypoints)

    @property
    def objects(self):
        if self.tracker is None:
            return []

        return self.tracker.objects

    def close(self):
        super().close()
        self.segmenter.close()
        self.tracker.close()

    def set_descriptor(self, desc):
        self.desc = desc
        for i, t in enumerate(self.templates):
            t.keypoints, t.features = self.desc.detectAndCompute(t.img, None)

    async def process_frame(self, frame, frame_ind):
        if(self._ready):
            #copy the frame into it so we don't have it processed by later methods
            asyncio.ensure_future(self._identify_features(frame, frame_ind))
        return

    async def decorate_frame(self, frame, name):

        if name != 'keypoints' and name != 'identify':
            return frame

        # draw key points
        for rect in self.segmenter.segments(frame):
            kp,_ = keypoints_view(self.desc, frame, rect)
            if(kp is not None):
                cv2.drawKeypoints(frame, kp, frame, color=(32,32,32), flags=0)
            # draw the rectangle that we use for kp
            rect = stretch_rectangle(rect, frame)
            draw_rectangle(frame, rect, (255, 0, 0), 1)

        if name == 'keypoints':
            return frame
        for n in self.features:
            for f in self.features[n]:
                color = f['color']
                kp = f['kp']
                kpcolor = f['kpcolor']
                for p,c in zip(kp, kpcolor):
                    cv2.circle(frame, tuple(p), 6, color, thickness=-1)

        return  frame#cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    async def _identify_features(self, frame, frame_ind):
        self._ready = False
        #make new features object
        features = {}

        found_feature = False
        for rect in self.segmenter.segments(frame):
            kp, des = keypoints_view(self.desc, frame, rect)
            if(des is not None and len(des) > 3):
                rect_features = await self._process_frame_view(frame, kp, des, rect, frame_ind)
                if len(rect_features) > 0:
                    found_feature = True
                    best = max(rect_features, key=lambda x: rect_features[x]['score'])
                    if best in features:
                        features[best].append(rect_features[best])
                    else:
                        features[best] = [rect_features[best]]
        if found_feature:
            self.features = features
        self._ready = True

    async def _process_frame_view(self, frame, kp, des, bounds, frame_ind):
        '''This method tries to run the calculation over multiple loops.
            The _ready is to in lieu of a callback on completion'''
        self._ready = False
        features = {}
        for t in self.templates:
            # check if t is already in play by its id number
            # if yes, check the frame index and see if this index is 2x the stride.
            templateInPlay = False
            for o in self.objects:
                if o['id'] == t.id:
                    templateInPlay = True

            if (templateInPlay and (frame_ind % (self.stride*2) != 0)):
                continue

            try:
                template = t.img
                name = t.label
                descriptors = t.keypoints, t.features

                if(type(descriptors[1]) != np.float32):
                    des1 = np.float32(descriptors[1])
                if(type(des) != np.float32):
                    des2 = np.float32(des)
                matches = self.matcher.knnMatch(des1, des2, k=2)
                # store all the good matches as per Lowe's ratio test.
                good = []
                if(len(matches) > 1): #not sure how this happens
                    for m,n in matches:
                        if m.distance < self.threshold * n.distance:
                            good.append(m)
                # check if we have enough good points
                if len(good) > self.min_match:

                    # look-up actual x,y keypoints
                    src_pts = np.float32([ descriptors[0][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                    dst_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                    # use homography to find matrix transform between them
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,3.0)
                    if M is None:
                        continue

                    src_poly = np.float32(t.poly).reshape(-1,1,2)
                    dst_poly = cv2.perspectiveTransform(src_poly,M)
                    dst_brect = cv2.boundingRect(dst_poly)

                    # check if the polygon is actually good
                    area = max(0.01, cv2.contourArea(dst_poly))
                    perimter = max(0.01, cv2.arcLength(dst_poly, True))
                    score = len(good) / len(des) * self.weights[0] + \
                            perimter / area * self.weights[1] + \
                            (dst_brect[2] / bounds[2] - 1 + dst_brect[3] /  bounds[3] - 1) * self.weights[2] + \
                            (dst_brect[2] * dst_brect[3] < 5) * self.weights[3] + \
                            self.weights[4]
                    if score > 0:
                        features[name] = { 'color': t.color, 'poly': np.int32(dst_poly),
                            'kp': np.int32([kp[m.trainIdx].pt for m in good]).reshape(-1,2),
                            'kpcolor': [(255, 255, 255, 128) for x in good],
                            'score': score, 'rect': bounds}
                        # register it with our tracker
                        if self.track:
                            self.tracker.track(frame, bounds, np.int32(dst_poly), name, t.id)
            except cv2.error:
                #not enough points
                await asyncio.sleep(0)
                continue

            #cede control
            await asyncio.sleep(0)
        return features