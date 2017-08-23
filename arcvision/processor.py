import asyncio
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from .contours import *

class Processor:
    def __init__(self, camera):
        camera.add_frame_processor(self)

class ExampleProcessor(Processor):
    '''Class that detects multiple objects given a set of labels and images'''
    def __init__(self, camera, width=500, height=500, stride=1):
        super().__init__(camera)
        self.corners = []
        self.corners.append((random.randrange(0, width / 4), random.randrange(0, height / 4)))
        self.corners.append((random.randrange(3 * width / 4, width), random.randrange(3 * height / 4, height)))
        self.color = tuple([random.randrange(0,255) for i in range(3)])
        self.stride = stride

    async def process_frame(self, frame, frame_ind):
        '''Perform update on frame, carrying out algorithm'''
        #simulate working hard
        return frame

    async def decorate_frame(self, frame):
        '''Draw visuals onto the given frame, without carrying-out update'''
        cv2.rectangle(frame, *self.corners, self.color, 2)
        return frame

class PreprocessProcessor(Processor):
    '''Substracts and computes background'''
    def __init__(self, camera, stride=1, bg_stride=16):
        super().__init__(camera)
        self.stride = stride

    async def process_frame(self, frame, frame_ind):
        '''Perform update on frame, carrying out algorithm'''
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    async def decorate_frame(self, frame):
        # go to BW but don't remove channel
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame

class BackgroundProcessor(Processor):
    '''Substracts and computes background'''
    def __init__(self, camera, stride=1, bg_stride=16):
        super().__init__(camera)
        self.stride = stride
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


    async def decorate_frame(self, frame):
        return frame * self.fg_mask[:,:,np.newaxis]

class TrackerProcessor(Processor):
    def __init__(self, camera, detector_stride, delete_threshold=0.2, stride=1):
        super().__init__(camera)
        self.tracking = []
        self.stride = stride
        self.names = {}
        # need to keep our own ticks because
        # we don't know frame index when track() is called
        self.ticks = 0
        self.min_obs_per_tick = self.stride / detector_stride * delete_threshold

    async def process_frame(self, frame, frame_ind):
        self.ticks += 1

        for i,t in enumerate(self.tracking[:]):
            status,bbox = t['tracker'].update(frame)
            #update polygon
            t['delta'][0] = bbox[0] - t['init'][0]
            t['delta'][1] = bbox[1] - t['init'][1]

            t['bbox'] = bbox
            # check obs counts
            if t['observed'] /  (self.ticks - t['start']) < self.min_obs_per_tick:
                del self.tracking[i]

        return frame


    async def decorate_frame(self, frame):
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
            cv2.putText(frame, '{}: {}'.format(t['id'], t['observed']), (0, 60 * (i+ 1)),
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
                if t['id'].split('-')[0] == name and TrackerProcessor._intersecting(bbox, t['bbox']):
                    # found existing one
                    # add to count
                    t['observed'] += 1
                    #update polygon and bounding box
                    t['poly'] = poly
                    t['init'] = bbox
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

class DetectionProcessor(Processor):
    def __init__(self, camera, query_images=[], labels=None, stride=10,
                 threshold=1.0, template_size=256, min_match=10):
        super().__init__(camera)
        #load template images
        if labels is None:
            labels = ['Key {}'.format(i) for i in range(len(query_images))]

        #read, convert to gray, and resize template images
        self.templates = []
        for k,i in zip(labels, query_images):
            d = {'name': k, 'path': i}
            img = cv2.imread(i, 0)
            h = int(template_size / img.shape[1] * img.shape[0])
            img = cv2.resize(img, (template_size, h))
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

        self.stride = stride
        self._ready = True
        self.features = {}
        self.threshold = threshold
        self.min_match = min_match

        '''
        # Initiate descriptors
        self.desc = cv2.AKAZE_create()
        #set-up our matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        '''
        # Initiate descriptors
        self.desc = cv2.BRISK_create()
        #set-up our matcher
        self.matcher = cv2.BFMatcher()

        #get descriptors for templates
        for t in self.templates:
            t['desc'] = self.desc.detectAndCompute(t['img'], None)
            if len(t['desc'][1]) == 0:
                raise ValueError('Unable to compute descriptors on {}'.format(t['path']))

        #set-up our tracker
        self.tracker = TrackerProcessor(camera, stride)



    async def process_frame(self, frame, frame_ind):
        if(self._ready):
            #copy the frame into it so we don't have it processed by later methods
            asyncio.ensure_future(self._identify_features(frame.copy()))
        return frame

    async def decorate_frame(self, frame):
        # draw key points
        kp,_ = self.desc.detectAndCompute(frame,None)
        cv2.drawKeypoints(frame, kp, frame, color=(32,32,32), flags=0)
        for n in self.features:
            for f in self.features[n]:
                points = f['poly']
                color = f['color']
                kp = f['kp']
                kpcolor = f['kpcolor']
                for p,c in zip(kp, kpcolor):
                    cv2.circle(frame, tuple(p), 6, color, thickness=-1)

                #draw polygon
                cv2.polylines(frame,[points],True,color, 3,cv2.LINE_AA)
                #get bottom of polygon
                y = sum([p[0][1] for p in points]) // 4
                x = sum([p[0][0] for p in points]) // 4
                cv2.putText(frame, n, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color)

        return frame

    async def _identify_features(self, frame):
        '''This method tries to run the calculation over multiple loops.
            The _ready is to in lieu of a callback on completion'''
        self._ready = False
        #make new features object
        features = {}

        #colormap
        cm = plt.cm.get_cmap()

        # find the keypoints and descriptors with desc
        kp2, des2 = self.desc.detectAndCompute(frame,None)
        if des2 is not None and len(des2) == 0: #check if we have descriptors
            return

        for t in self.templates:
            features[t['name']] = []
        for t in self.templates:
            template = t['img']
            name = t['name']
            descriptors = t['desc']
            h,w, = template.shape

            '''
            matches = self.matcher.match(descriptors[1],des2)
            matches = sorted(matches, key = lambda x:x.distance)

            good = matches[:self.min_match]
            src_pts = np.float32([ descriptors[0][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,2.0)
            matchesMask = mask.ravel().tolist()

            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            features[name].append({ 'color': t['color'], 'poly': np.int32(dst),
                                    'kp': np.int32([kp2[m.trainIdx].pt for m in good]).reshape(-1,2),
                                    'kpcolor': [cm(x.distance / good[-1].distance) for x in good]})

            '''
            matches = self.matcher.knnMatch(descriptors[1], des2, k=2)
            # store all the good matches as per Lowe's ratio test.
            good = []
            if(len(matches[0]) > 1): #not sure how this happens
                for m,n in matches:
                    if m.distance < self.threshold * n.distance:
                        good.append(m)

            # check if we have enough good points
            await asyncio.sleep(0)
            if len(good) > self.min_match:
                # look-up actual x,y keypoints
                src_pts = np.float32([ descriptors[0][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                # use homography to find matrix transform between them
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()

                #pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                pts = np.float32([t['poly']]).reshape(-1,1,2)
                try:
                    dst = cv2.perspectiveTransform(pts,M)
                except cv2.error:
                    #not enough points
                    await asyncio.sleep(0)
                    continue

                await asyncio.sleep(0)
                # check if the polygon is actually good
                area = cv2.contourArea(dst)
                perimter = cv2.arcLength(dst, True)
                if(cv2.isContourConvex(dst) and area / perimter > 1):
                    features[name].append({ 'color': t['color'], 'poly': np.int32(dst),
                        'kp': np.int32([kp2[m.trainIdx].pt for m in good]).reshape(-1,2),
                        'kpcolor': [cm(x.distance / good[-1].distance) for x in good]})
                    # register it with our tracker
                    self.tracker.track(frame, np.int32(dst), t['name'])

            #cede control
            await asyncio.sleep(0)
        self._ready = True
        #now swap with our features
        self.features = features
