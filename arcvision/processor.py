import asyncio
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from .contours import *

class Processor:
    def __init__(self, camera, streams, stride):
        camera.add_frame_processor(self)
        self.streams = streams
        self.stride = stride

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
    def __init__(self, camera, bg_stride=16):
        super().__init__(camera, ['background-removal'], 1)
        self.mog = cv2.createBackgroundSubtractorMOG2(5000)
        self.bg_stride = bg_stride
        self.fg_mask = None
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    async def process_frame(self, frame, frame_ind):
        '''Perform update on frame, carrying out algorithm'''
        if self.fg_mask is None:
            self.fg_mask = np.uint8(frame.copy())

        if frame_ind % self.bg_stride == 0:
            self.mog.apply(frame, self.fg_mask)
            # self.fg_mask = cv2.morphologyEx(self.fg_mask, cv2.MORPH_OPEN, self.kernel)
        return frame * self.fg_mask


    async def decorate_frame(self, frame, name):
        if(len(frame.shape) == 3):
            return frame * self.fg_mask
        else:
            return frame * self.fg_mask[:,:,np.newaxis]

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
    def __init__(self, camera, stride):
        super().__init__(camera, ['background', 'distance', 'markers', 'watershed', 'boxes'], stride)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.rect_iter = range(0)

    async def process_frame(self, frame, frame_ind):
        '''we only process on request'''
        return frame

    def _process_frame(self, frame, frame_ind):
        bg = self.filter_background(frame)
        dist_transform = self.filter_distance(bg)
        markers = self.filter_ws_markers(dist_transform)
        ws_markers = cv2.watershed(frame, markers)
        self.rect_iter = self.watershed(frame, markers)
        return frame

    def segments(self, frame = None):
        if frame is not None:
            self._process_frame(frame, 0)
        yield from self.rect_iter

    def filter_background(self, frame):
        #img = cv2.pyrMeanShiftFiltering(frame, 21, 51)
        img = cv2.blur(frame, (3,3))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, bg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # noise removal
        kernel = np.ones((3,3),np.uint8)
        bg = cv2.erode(bg, kernel, iterations = 2)
        bg = cv2.morphologyEx(bg,cv2.MORPH_OPEN,kernel, iterations = 2)
        # check if perhaps our colors are inverted. background is assumed to be largest
        if(np.mean(bg) > 255 / 2):
            bg[bg == 255] = 5
            bg[bg == 0] = 255
            bg[bg == 5] = 0
        return bg

    def filter_distance(self, frame):
        dist_transform = cv2.distanceTransform(frame, cv2.DIST_L2,0)
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

    def watershed(self, frame, markers):
        ws_markers = cv2.watershed(frame, markers)
        for i in range(1, np.max(ws_markers)):
            pixels = np.argwhere(ws_markers == i)
            rect = cv2.boundingRect(pixels)
            #flip around our rectangle
            rect = (rect[1], rect[0], rect[3], rect[2])
            # exempt small or large rectangles (> 25 % of screen)
            if(len(pixels) < 5 or rect[2] * rect[3] < 20 or rect[2] * rect[3] / frame.shape[0] / frame.shape[1] > 0.25 ):
                continue
            yield rect


    async def decorate_frame(self, frame, name):
        bg = self.filter_background(frame)
        if name == 'background':
            return bg

        dist_transform = self.filter_distance(bg)
        if name == 'distance':
            return dist_transform

        markers = self.filter_ws_markers(dist_transform)
        if name == 'markers':
            return markers * 10000

        if name == 'watershed':
            ws_markers = cv2.watershed(frame, markers)
            frame[ws_markers == -1] = (255, 255, 0)
            return frame

        for rect in self.watershed(frame, markers):
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 255, 0), 1)
        return frame


class DetectionProcessor(Processor):
    '''Detects query images in frame. Uses async to spread out computation. Cannot handle replicas of an object in frame'''
    def __init__(self, camera, query_images=[], labels=None, stride=10,
                 threshold=1.0, template_size=256, min_match=6, weights=[1, 2]):

        #we have a specific order required
        #set-up our tracker
        self.tracker = _TrackerProcessor(camera, stride)
        #then our segmenter
        self.segmenter = _SegmentProcessor(camera, stride)
        #then us
        super().__init__(camera, ['keypoints', 'identify'], stride)
        #load template images
        if labels is None:
            labels = ['Key {}'.format(i) for i in range(len(query_images))]

        #read, convert to gray, and resize template images
        self.templates = []
        for k,i in zip(labels, query_images):
            d = {'name': k, 'path': i}
            img = cv2.imread(i, 0)
            #h = int(template_size / img.shape[1] * img.shape[0])
            #img = cv2.resize(img, (template_size, h))
            # get contours
            processed, poly = find_template_contour(img)
            # write it out so we can double check
            cv2.imwrite(i.split('.jpg')[0] + '_contours.jpg', processed)
            d['img'] = processed
            d['poly'] = poly
            self.templates.append(d)

        #create color gradient
        N = len(labels)
        cm = plt.cm.get_cmap('Dark2')
        for i,t in enumerate(self.templates):
            rgba = cm(i / N)
            rgba = [int(x * 255) for x in rgba]
            t['color'] = rgba[:-1]

        self._ready = True
        self.features = {}
        self.threshold = threshold
        self.min_match = min_match
        self.weights = weights

        # Initiate descriptors
        self.desc = cv2.KAZE_create()
        #set-up our matcher
        self.matcher = cv2.BFMatcher()

        #get descriptors for templates
        for t in self.templates:
            t['desc'] = self.desc.detectAndCompute(t['img'], None)
            if t['desc'] is None or len(t['desc'][1]) == 0:
                raise ValueError('Unable to compute descriptors on {}'.format(t['path']))

    async def process_frame(self, frame, frame_ind):
        if(self._ready):
            #copy the frame into it so we don't have it processed by later methods
            asyncio.ensure_future(self._identify_features(frame.copy()))
        return frame

    async def decorate_frame(self, frame, name):
        # draw key points
        for rect in self.segmenter.segments(frame):
            kp,_ = self._get_keypoints(frame, rect)
            cv2.drawKeypoints(frame, kp, frame, color=(32,32,32), flags=0)
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
                y = sum([p[0][1] for p in points]) // 4
                x = sum([p[0][0] for p in points]) // 4
                cv2.putText(frame, '{} ({:.2})'.format(n, f['score']), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color)

        return frame

    def _get_keypoints(self, frame, rect):
        '''return the keypoints limited to a region'''

        frame_view = frame[ rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2]) ]
        kp, des = self.desc.detectAndCompute(frame_view,None)
        # need to transform the key points back
        for i in range(len(kp)):
            kp[i].pt = (rect[0] + kp[i].pt[0], rect[1] + kp[i].pt[1])
        return kp, des

    async def _identify_features(self, frame):
        self._ready = False
        #make new features object
        features = {}

        for rect in self.segmenter.segments(frame):
            kp, des = self._get_keypoints(frame, rect)
            if(des is not None and len(des) > 2):
                features = await self._process_frame_view(features, frame, kp, des)

        #now swap with our features
        self.features = features
        self._ready = True

    async def _process_frame_view(self, features, frame, kp, des):
        '''This method tries to run the calculation over multiple loops.
            The _ready is to in lieu of a callback on completion'''
        self._ready = False

        for t in self.templates:
            features[t['name']] = []
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
                    matchesMask = mask.ravel().tolist()

                    src_poly = np.float32([t['poly']]).reshape(-1,1,2)
                    dst_poly = cv2.perspectiveTransform(src_poly,M)
                    dst_bbox = cv2.boundingRect(dst_poly)

                    # check if the polygon is actually good
                    area = cv2.contourArea(dst_poly)
                    perimter = max(0.01, cv2.arcLength(dst_poly, True))
                    if cv2.isContourConvex(dst_poly) and area / perimter > 0.0:
                        score = len(good) * self.weights[0] + area / perimter * self.weights[1]
                        cm = plt.cm.get_cmap()
                        features[name].append({ 'color': t['color'], 'poly': np.int32(dst_poly),
                            'kp': np.int32([kp[m.trainIdx].pt for m in good]).reshape(-1,2),
                            'kpcolor': [cm(x.distance / good[-1].distance) for x in good],
                            'score': score})
                        # register it with our tracker
                        self.tracker.track(frame, np.int32(dst_poly), t['name'])
            except cv2.error:
                #not enough points
                await asyncio.sleep(0)
                continue

            #cede control
            await asyncio.sleep(0)
        return features
