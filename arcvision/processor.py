import asyncio
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt


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
        return frame


class TrackerProcessor:
    def __init__(self):
        self.tracking = []

    async def process_frame(self, frame):
        pass

    async def decorate_frame(self, frame):
        return await self.process_frame(frame)


class DetectionProcessor:
    def __init__(self, query_images=[], labels=None, stride=10, threshold=1.0, template_size=256, min_match=10):
        #load template images
        if labels is None:
            labels = ['Key {}'.format(i) for i in range(len(query_images))]

        #read, convert to gray, and resize template images
        self.templates = []
        for k,i in zip(labels, query_images):
            d = {'name': k, 'path': i}
            img = cv2.imread(i, 0)
            h = int(template_size / img.shape[1] * img.shape[0])
            print(h)
            img = cv2.resize(img, (template_size, h))
            d['img'] = img
            self.templates.append(d)

        #create color gradient
        N = len(labels)
        cm = plt.cm.get_cmap('Dark2')
        for i,t in enumerate(self.templates):
            rgba = cm(i / N)
            rgba = [int(x * 255) for x in rgba]
            t['color'] = rgba[:-1]

        self.stride = 1
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
        self.desc = cv2.AKAZE_create()
        #set-up our matcher
        self.matcher = cv2.BFMatcher()


        #get descriptors for templates
        for t in self.templates:
            t['desc'] = self.desc.detectAndCompute(t['img'], None)
            if len(t['desc'][1]) == 0:
                raise ValueError('Unable to compute descriptors on {}'.format(t['path']))




    async def process_frame(self, frame):
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
        if len(des2) == 0: #check if we have descriptors
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

                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,M)

                await asyncio.sleep(0)
                # check if the polygon is actually good
                area = cv2.contourArea(dst)
                perimter = cv2.arcLength(dst, True)
                if(cv2.isContourConvex(dst) and area / perimter > 1):
                    features[name].append({ 'color': t['color'], 'poly': np.int32(dst),
                        'kp': np.int32([kp2[m.trainIdx].pt for m in good]).reshape(-1,2),
                        'kpcolor': [cm(x.distance / good[-1].distance) for x in good]})


            #cede control
            await asyncio.sleep(0)
        self._ready = True
        #now swap with our features
        self.features = features
