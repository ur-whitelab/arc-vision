import asyncio
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from .contours import *

class Processor:
    def __init__(self, camera, streams, stride):

        self.streams = streams
        self.stride = stride
        camera.add_frame_processor(self)

class CropProcessor(Processor):
    '''Class that detects multiple objects given a set of labels and images'''
    def __init__(self, camera, rect, stride=1):
        super().__init__(camera, ['crop'], stride)
        self.rect = [ (rect[0], rect[1]), (rect[2], rect[3]) ]
        self.mask = None
        self.dmask = None

    async def process_frame(self, frame, frame_ind):
        '''Perform update on frame, carrying out algorithm'''
        if self.mask is None:
            self.mask = np.zeros(frame.shape, dtype=np.uint8)
            cv2.rectangle(self.mask, *self.rect, (255,)*3, thickness=cv2.FILLED)
        #frame = cv2.bitwise_and(frame, self.mask)
        frame = frame[ self.rect[0][1]:self.rect[1][1], self.rect[0][0]:self.rect[1][0]]
        return frame

    async def decorate_frame(self, frame, name):
        '''Draw visuals onto the given frame, without carrying-out update'''
        if self.dmask is None:
            self.dmask = np.zeros(frame.shape, dtype=np.uint8)
            cv2.rectangle(self.dmask, *self.rect, (255,255,255), thickness=cv2.FILLED)
        #frame = cv2.bitwise_and(frame, self.dmask)
        frame = frame[ self.rect[0][1]:self.rect[1][1], self.rect[0][0]:self.rect[1][0],:  ]
        return frame

class PreprocessProcessor(Processor):
    '''Substracts and computes background'''
    def __init__(self, camera, stride=1, gamma=2.5):
        super().__init__(camera, ['preprocess'], stride)


    async def process_frame(self, frame, frame_ind):
        '''Perform update on frame, carrying out algorithm'''
        #return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    async def decorate_frame(self, frame, name):
        # go to BW but don't remove channel
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame


class BackgroundProcessor(Processor):
    '''Substracts and computes background'''
    def __init__(self, camera, countdown=30):
        super().__init__(camera, ['background-removal'], 1)
        self.avg_background = None
        self.countdown = countdown
        self.countdown_start = countdown

    async def process_frame(self, frame, frame_ind):
        '''Perform update on frame, carrying out algorithm'''

        if self.avg_background is None:
            self.avg_background = np.empty(frame.shape, dtype=np.uint32)
            self.avg_background[:] = 0

        if self.countdown > 0:
            self.avg_background += frame
            self.countdown -= 1
            if self.countdown == 0:
                self.avg_background //= self.countdown_start

                self.avg_background = self.avg_background.astype(np.uint8)

                kernel = np.ones((5,5),np.uint8)
                self.avg_background = cv2.morphologyEx(self.avg_background,cv2.MORPH_OPEN,kernel, iterations = 2)



        return frame


    async def decorate_frame(self, frame, name):

        if self.countdown > 0:
            frame = self.avg_background.copy().astype(np.uint8)
            cv2.putText(frame,
                        'Processing Background - {}'.format(self.countdown),
                        (0, frame.shape[1] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

        return frame

class _TrackerProcessor(Processor):
    def __init__(self, camera, detector_stride, delete_threshold=0.2, stride=1):
        super().__init__(camera, ['track'], stride)
        self.tracking = []
        self.names = {}
        # need to keep our own ticks because
        # we don't know frame index when track() is called
        self.ticks = 0
        self.min_obs_per_tick = self.stride / detector_stride * delete_threshold

    async def process_frame(self, frame, frame_ind):
        self.ticks += 1
        delete = []
        for i,t in enumerate(self.tracking):
            status,bbox = t['tracker'].update(frame)
            #update polygon
            if(status):
                t['delta'][0] = bbox[0] - t['init'][0]
                t['delta'][1] = bbox[1] - t['init'][1]
                t['bbox'] = bbox

            # check obs counts
            if t['observed'] /  (self.ticks - t['start']) < self.min_obs_per_tick:
                delete.append(i)
        offset = 0
        delete.sort()
        for i in delete:
            del self.tracking[i - offset]
            offset += 1

        return frame


    async def decorate_frame(self, frame, name):
        for i,t in enumerate(self.tracking):
            if(t['observed'] < 3):
                continue
            bbox = t['bbox']
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0,0,255), 1)

            #now draw polygon
            cv2.polylines(frame,[t['poly'] + t['delta']], True, (0,0,255), 3)

            #put note about it
            cv2.putText(frame,
                        '{}: {:.2} (limit: {:.2})'.format(t['id'], t['observed'] /  (self.ticks - t['start']), self.min_obs_per_tick ),
                        (0, 60 * (i+ 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))

        return frame

    def _intersecting(a, b, threshold=0.25):
        dx = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
        dy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
        if (dx >= 0) and (dy >= 0):
            # check if most of one square's area is included
            intArea = dx * dy
            minArea = min(a[1] * a[3],  b[1] * b[3])
            if(intArea / minArea > threshold):
              return True
        return False

    def track(self, frame, poly, name):

        bbox = cv2.boundingRect(poly)

        if name in self.names:
            #we need to make sure we don't have an existing example
            for t in self.tracking:
                if t['id'].split('-')[0] == name and _TrackerProcessor._intersecting(bbox, t['bbox']):
                    # found existing one
                    # add to count
                    t['observed'] += 1
                    #update polygon and bounding box
                    t['poly'] = poly
                    t['init'] = bbox
                    t['delta'] = np.int32([0,0])
                    t['tracker'].init(frame, bbox)
                    return False
            id = '{}-{}'.format(name, self.names[name])
            self.names[name] += 1
        else:
            self.names[name] = 0
            id = '{}-{}'.format(name, self.names[name])

        tracker = cv2.TrackerMedianFlow_create()
        status = tracker.init(frame, bbox)

        if not status:
            print('Failed to initialize tracker')
            return False

        track_obj = {'id': id,
                     'tracker': tracker,
                     'poly': poly,
                     'init': bbox,
                     'bbox': bbox,
                     'observed': 1,
                     'start': self.ticks,
                     'delta': np.int32([0,0])}
        self.tracking.append(track_obj)
        return True

class _SegmentProcessor(Processor):
    def __init__(self, camera, stride, max_segments):
        super().__init__(camera, ['background-subtract', 'background-thresh', 'background-dilate', 'background-open', 'background', 'distance', 'boxes'], stride)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.rect_iter = range(0)
        self.background_processor = BackgroundProcessor(camera)
        self.mode = 'preprocess'
        self.max_segments = max_segments

    async def process_frame(self, frame, frame_ind):
        '''we only process on request'''
        return frame

    def _process_frame(self, frame, frame_ind):
        bg = self.filter_background(frame)
        dist_transform = self.filter_distance(bg)
        self.rect_iter = self.filter_contours(dist_transform)
        return frame

    def segments(self, frame = None):
        if frame is not None:
            self._process_frame(frame, 0)
        yield from self.rect_iter

    def filter_background(self, frame, name = ''):
        #img = cv2.pyrMeanShiftFiltering(frame, 21, 51)

        img = frame.copy()
        if self.mode == 'production':
            img -= self.background_processor.avg_background

        if name == 'background-subtract':
            return img
        if self.mode == 'production':
            img = cv2.blur(img, (6,6))
        else:
            img = cv2.blur(img, (3,3))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, bg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        #bg = th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                cv2.THRESH_BINARY_INV,11,2)
        if name == 'background-thresh':
            return bg
        # noise removal
        kernel = np.ones((3,3),np.uint8)
        if(self.mode == 'preprocess'):
            bg = cv2.erode(bg, kernel, iterations = 2)
        else:
            bg = cv2.dilate(bg, kernel, iterations = 4)
        if name == 'background-dilate':
            return bg
        bg = cv2.morphologyEx(bg,cv2.MORPH_OPEN,kernel, iterations = 2)
        if name == 'background-open':
            return bg

        # check if perhaps our colors are inverted. background is assumed to be largest
        if(np.mean(bg) > 255 / 2):
            bg[bg == 255] = 5
            bg[bg == 0] = 255
            bg[bg == 5] = 0
        return bg

    def filter_distance(self, frame):
        if self.mode == 'preprocess':
            dist_transform = cv2.distanceTransform(frame, cv2.DIST_L2,0)
        else:
            dist_transform = cv2.distanceTransform(frame, cv2.DIST_L2,5)
        dist_transform = cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)

        #create distance tranform contours
        dist_transform = np.uint8(dist_transform)
        return dist_transform

    def filter_ws_markers(self, frame):
        _, contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #create markers
        markers = np.zeros( frame.shape, dtype=np.uint8 )

        for i in range(len(contours)):
            #we draw onto our markers with fill to create the mask
            cv2.drawContours(markers, contours, i, (i + 1,), -1)
        #draw a tiny circle to indicate background hint
        cv2.circle(markers, (5,5), 3, (255,))
        return markers.astype(np.int32)

    def filter_contours(self, frame):
        _, contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(c) for c in contours]
        rects.sort(key = lambda r: r[2] * r[3], reverse=True)
        segments = 0

        for r in rects:
            #flip around our rectangle
            # exempt small or large rectangles (> 25 % of screen)
            if(r[2] * r[3] < 250 or r[2] * r[3] / frame.shape[0] / frame.shape[1] > 0.25 ):
                continue
            yield r
            segments += 1
            if(segments == self.max_segments):
                break


    def watershed(self, frame, markers):
        ws_markers = cv2.watershed(frame, markers)
        segments = 0
        for i in range(1, np.max(ws_markers)):
            pixels = np.argwhere(ws_markers == i)
            rect = cv2.boundingRect(pixels)
            #flip around our rectangle
            rect = (rect[1], rect[0], rect[3], rect[2])
            # exempt small or large rectangles (> 25 % of screen)
            if(len(pixels) < 5 or rect[2] * rect[3] < 100 or rect[2] * rect[3] / frame.shape[0] / frame.shape[1] > 0.25 ):
                continue
            yield rect
            segments += 1
            if(segments == self.max_segments):
                break

    def polygon(self, frame):
        bg = self.filter_background(frame)
        dist_transform = self.filter_distance(bg)
        markers = self.filter_ws_markers(dist_transform)
        ws_markers = cv2.watershed(frame, markers)

        #sort based on size
        pixels = [np.flip(np.argwhere(ws_markers == i), axis=1) for i in range(1, np.max(ws_markers))]
        def key(x):
            r = cv2.boundingRect(x)
            return r[2] * r[3]
        pixels.sort(key = key, reverse=True)
        for p in pixels:
            # exempt small rectangles
            rect = cv2.boundingRect(p)
            if(len(p) < 5 or rect[2] * rect[3] < 20 ):
                continue
            # once we find one, use it
            hull = cv2.convexHull(p)
            return frame, hull, rect

        print('Could not identify polygon. See processed file.')
        return frame, frame.shape[:2]

    async def decorate_frame(self, frame, name):
        bg = self.filter_background(frame, name)
        if name.find('background') != -1:
            return bg

        dist_transform = self.filter_distance(bg)
        if name == 'distance':
            return dist_transform

        for rect in self.filter_contours(dist_transform):
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 255, 0), 1)
        return frame


class DetectionProcessor(Processor):
    '''Detects query images in frame. Uses async to spread out computation. Cannot handle replicas of an object in frame'''
    def __init__(self, camera, query_images=[], labels=None, stride=10,
                 threshold=1.0, template_size=256, min_match=8, weights=[3, 0.2, -10, 1], max_segments=10):

        #we have a specific order required
        #set-up our tracker
        # give estimate of our stride
        self.tracker = _TrackerProcessor(camera, stride * 2 * len(query_images))
        #then our segmenter
        self.segmenter = _SegmentProcessor(camera, stride, max_segments)
        #then us
        super().__init__(camera, ['keypoints', 'identify'], stride)


        self._ready = True
        self.features = {}
        self.threshold = threshold
        self.min_match = min_match
        self.weights = weights
        self.stretch_boxes=1.5


        # Initiate descriptors
        self.desc = cv2.KAZE_create()
        #set-up our matcher
        self.matcher = cv2.BFMatcher()

        #load template images
        if labels is None:
            labels = ['Key {}'.format(i) for i in range(len(query_images))]
        #read, convert to gray, and resize template images
        self.templates = []
        for k,i in zip(labels, query_images):
            t = {'name': k, 'path': i}
            img = cv2.imread(i)
            #h = int(template_size / img.shape[1] * img.shape[0])
            #img = cv2.resize(img, (template_size, h))
            # get contours
            processed, poly, rect = self.segmenter.polygon(img)
            t['desc'] = self._get_keypoints(img, rect)
            if t['desc'][1] is None or len(t['desc'][1]) == 0:
                raise ValueError('Unable to compute descriptors on {}'.format(t['path']))
            # put info on image
            cv2.polylines(processed, [poly], True, (0,0,255), 3)
            cv2.drawKeypoints(processed, t['desc'][0], processed, color=(32,32,32), flags=0)
            # write it out so we can double check
            cv2.imwrite(i.split('.jpg')[0] + '_contours.jpg', processed)
            t['img'] = processed
            t['poly'] = poly
            self.templates.append(t)

        #create color gradient
        N = len(labels)
        cm = plt.cm.get_cmap('Dark2')
        for i,t in enumerate(self.templates):
            rgba = cm(i / N)
            rgba = [int(x * 255) for x in rgba]
            t['color'] = rgba[:-1]


        #switch out of prerocess mode
        self.segmenter.mode = 'production'



    async def process_frame(self, frame, frame_ind):
        if(self._ready):
            #copy the frame into it so we don't have it processed by later methods
            asyncio.ensure_future(self._identify_features(frame.copy()))
        return frame

    async def decorate_frame(self, frame, name):

        if name != 'keypoints' and name != 'identify':
            return frame

        # draw key points
        for rect in self.segmenter.segments(frame):
            kp,_ = self._get_keypoints(frame, rect)
            if(kp is not None):
                cv2.drawKeypoints(frame, kp, frame, color=(32,32,32), flags=0)
            # draw the rectangle that we use for kp
            rect = self._stretch_rectangle(rect, frame)
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 1)

        if name == 'keypoints':
            return frame
        for n in self.features:
            for f in self.features[n]:
                points = f['poly']
                color = f['color']
                kp = f['kp']
                kpcolor = f['kpcolor']
                for p,c in zip(kp, kpcolor):
                    cv2.circle(frame, tuple(p), 6, color, thickness=-1)

                #draw polygon
                cv2.polylines(frame,[points],True, color, 3, cv2.LINE_AA)
                #get bottom of polygon
                cv2.putText(frame, '{} ({:.2})'.format(n, f['score']), (f['rect'][0], f['rect'][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color)

        return frame

    def _stretch_rectangle(self, rect, frame):
        # stretch out the rectangle
        rect = list(rect)
        rect[0] += int(rect[2] * (1 - self.stretch_boxes) // 2)
        rect[1] += int(rect[3] * (1 - self.stretch_boxes) // 2)
        rect[2] = int(rect[2] * self.stretch_boxes)
        rect[3] = int(rect[3] * self.stretch_boxes)

        rect[0] = max(rect[0], 0)
        rect[1] = max(rect[1], 0)
        rect[2] = min(frame.shape[1], rect[2])
        rect[3] = min(frame.shape[0], rect[3])
        return rect

    def _get_keypoints(self, frame, rect):
        '''return the keypoints limited to a region'''
        rect = self._stretch_rectangle(rect, frame)

        frame_view = frame[ rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2]) ]
        kp, des = self.desc.detectAndCompute(frame_view,None)
        #need to transform the key points back
        for i in range(len(kp)):
            kp[i].pt = (rect[0] + kp[i].pt[0], rect[1] + kp[i].pt[1])
        return kp, des

    async def _identify_features(self, frame):
        self._ready = False
        #make new features object
        features = {}

        found_feature = False
        for rect in self.segmenter.segments(frame):
            kp, des = self._get_keypoints(frame, rect)
            if(des is not None and len(des) > 2):
                rect_features = await self._process_frame_view(frame, kp, des, rect)
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

    async def _process_frame_view(self, frame, kp, des, bounds):
        '''This method tries to run the calculation over multiple loops.
            The _ready is to in lieu of a callback on completion'''
        self._ready = False
        features = {}
        for t in self.templates:
            try:
                template = t['img']
                name = t['name']
                descriptors = t['desc']

                matches = self.matcher.knnMatch(descriptors[1], des, k=2)
                # store all the good matches as per Lowe's ratio test.
                good = []
                if(len(matches[0]) > 1): #not sure how this happens
                    for m,n in matches:
                        if m.distance < self.threshold * n.distance:
                            good.append(m)
                # check if we have enough good points
                if len(good) > self.min_match:

                    # look-up actual x,y keypoints
                    src_pts = np.float32([ descriptors[0][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                    dst_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                    # use homography to find matrix transform between them
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

                    src_poly = np.float32(t['poly']).reshape(-1,1,2)
                    dst_poly = cv2.perspectiveTransform(src_poly,M)
                    dst_bbox = cv2.boundingRect(dst_poly)

                    # check if the polygon is actually good
                    area = cv2.contourArea(dst_poly)
                    perimter = max(0.01, cv2.arcLength(dst_poly, True))
                    score = len(good) / len(descriptors[0]) * self.weights[0] + \
                            area / perimter * self.weights[1] + \
                            (dst_bbox[2] > bounds[2] or dst_bbox[3] >  bounds[3]) * self.weights[2] + \
                            self.weights[3]
                    if score > 0:
                        cm = plt.cm.get_cmap()
                        features[name] = { 'color': t['color'], 'poly': np.int32(dst_poly),
                            'kp': np.int32([kp[m.trainIdx].pt for m in good]).reshape(-1,2),
                            'kpcolor': [cm(x.distance / good[-1].distance) for x in good],
                            'score': score, 'rect': bounds}
                        # register it with our tracker
                        # self.tracker.track(frame, np.int32(dst_poly), t['name'])
            except cv2.error:
                #not enough points
                await asyncio.sleep(0)
                continue

            #cede control
            await asyncio.sleep(0)
        return features
