import asyncio
import random
import sys
import cv2
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from .utils import *

_object_id = 0

def object_id():
    global _object_id
    _object_id += 1
    return _object_id

class Processor:
    def __init__(self, camera, streams, stride):

        self.streams = streams
        self.stride = stride
        camera.add_frame_processor(self)
        self.camera = camera

    @property
    def objects(self):
        return []

    def close(self):
        self.camera.remove_frame_processor(self)


class StrobeProcessor(Processor):
    ''' This will tune camera parameters to optimize strobe'''
    def __init__(self, camera, background, stride=1):
        super().__init__(camera, ['strobe'], stride)
        # these starting points should work
        self.strobe_args = [1 / 60, 1 / 60, 10, 10]
        self.proposed_args = self.strobe_args[:]
        self.strobe_steps = [1 / 60, 0, 1, 1]
        self.best_args = self.strobe_args[:]
        self.best_score = 100
        self._prob = 0.5
        self.camera.set_strobe_args(*self.strobe_args, 0.4)
        self.camera.start_strobe()
        self.background = background
        self.index_start = camera.frame_ind

    def close(self):
        self.camera.set_strobe_args(*self.best_args, 0.2)
        super().close()

    def _arg_score(self):
        return self.strobe_args[1] * 30 * 5 + self.strobe_args[3]
        #return sum([x / y for x,y in zip(self.strobe_args, self.strobe_steps)])

    def _update(self, args, scale=1):
        scale = int(scale)
        for i in range(len(args)):
            if random.random() < self._prob:
                args[i] += self.strobe_steps[i] *  int(random.randint(-scale, scale))
                # none can be less than 0
                args[i] = max(0, args[i])
                # input delay cannot be more than 15 frames
                if i == 0:
                    args[0] = min(15 / 60, args[0])
        self.camera.set_strobe_args(*self.strobe_args, 0.4)

    def _is_strobed(self, frame):
        # look for green.
        print(np.mean(frame - self.background, axis=(0,1))[1] > 150, np.mean(frame - self.background, axis=(0,1))[1])
        return np.mean(frame - self.background, axis=(0,1))[1] > 150

    async def process_frame(self, frame, frame_ind):
        ''' some quasi monte carlo'''
        self.success = self._is_strobed(frame)
        if(not self.success):
            # randomly accept bad one
            if random.random() < 0.3:
                self.strobe_args = self.proposed_args
        else:
            # good params, check if best
            if(self._arg_score() < self.best_score):
                self.best_args = self.strobe_args[:]
                self.best_score = self._arg_score()
            self.strobe_args = self.proposed_args

        args = self.strobe_args[:]
        scale = 20 / (frame_ind + 1 - self.index_start) # add 1 for start / 0
        self._update(args, max(1, scale))
        self.proposed_args = args
        return frame

    async def decorate_frame(self, frame, name):
        if name == 'strobe':
            cv2.putText(frame,
                    'Current: {}'.format(self.strobe_args),
                    (100, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,124,255))
            cv2.putText(frame,
                    'Best Strobe: {} ({})'.format(self.best_args, self.best_score),
                    (0, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255))
            cv2.putText(frame, 'Status: {}'.format(self.success), (0, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255))
        return frame

class CalibrationProcessor(Processor):
    '''This will find a perspective transform that goes from our coordinate system
       to the projector coordinate system. '''
    def __init__(self, camera, background, stride=1, N=8, delay=2, stay=3):
        self.segmenter = SegmentProcessor(camera, background, -1, 1, max_rectangle=0.1)
        super().__init__(camera, ['calibration', 'transform'], stride)

        self.calibration_points = np.random.random( (N, 2)) * 0.8 + 0.1
        print(self.calibration_points)
        self._objects = []
        for i,c in enumerate(self.calibration_points):
            o = {}
            o['id'] = object_id()
            o['center_scaled'] = c
            o['label'] = 'calibration-point'
            self._objects.append(o)

        self.index = 0
        self.delay = delay
        self.stay = stay
        self.points = np.zeros( (N, 2) )
        self.counts = np.zeros( (N, 1) )
        self.N = N
        self.first = True

        self.transform = np.identity(3)

    def close(self):
        super().close()
        self.segmenter.close()

    async def process_frame(self, frame, frame_ind):
        try:
            if frame_ind % (self.stay + self.delay) > self.delay:
                seg = next(self.segmenter.segments(frame))
                p = rect_scaled_center(seg, frame)
                self.points[self.index] += p
                self.counts[self.index] += 1
        except StopIteration:
            pass

        if frame_ind % (self.stay + self.delay) == 0:
            # update homography estimate
            if self.index == self.N - 1:
                print('updating homography')
                self._update_homography(frame)
                print(self.transform)
                self.points[:] = 0
                self.counts[:] = 0
            self.index += 1
            self.index %= self.N

        return frame


    def _update_homography(self, frame):
        t, _ = cv2.findHomography(self.points / self.counts,
                                self.calibration_points,
                                 cv2.LMEDS)
        if t is None:
            print('homography failed')
        else:
            if(self.first):
                self.transform = t
                self.first = True
            else:
                self.transform = self.transform * 0.7 + t * 0.3

    def _unscale(self, array, shape):
        return (array * shape[:2]).astype(np.int32)
    async def decorate_frame(self, frame, name):
        if name == 'calibration' or name == 'transform':
            p = self.calibration_points[self.index, :]
            c = cv2.perspectiveTransform(p.reshape(-1, 1, 2), linalg.inv(self.transform)).reshape(2)
            c *= frame.shape[:2]
            cv2.circle(frame,
                        tuple(c.astype(np.int)), 10, (255,0,0), -1)
            cv2.circle(frame,
                        tuple(self._unscale(self.points[self.index] / self.counts[self.index],
                            frame.shape)), 10, (0,0,255), -1)
        if name == 'transform':
            for i in range(frame.shape[2]):
                frame[:,:,i] = cv2.warpPerspective(frame[:,:,i],
                                                    self.transform,
                                                     frame.shape[1::-1])
        return frame


    @property
    def objects(self):
        return [self._objects[self.index]]

class CropProcessor(Processor):
    '''Class that detects multiple objects given a set of labels and images'''
    def __init__(self, camera, rect=None, stride=1):
        super().__init__(camera, ['crop'], stride)
        self.rect = rect

    async def process_frame(self, frame, frame_ind):
        '''Perform update on frame, carrying out algorithm'''
        if self.rect is not None:
            return rect_view(frame, self.rect)
        return frame

    async def decorate_frame(self, frame, name):
        '''Draw visuals onto the given frame, without carrying-out update'''
        return await self.process_frame(frame, 0)

class PreprocessProcessor(Processor):
    '''Substracts and computes background'''
    def __init__(self, camera, stride=1, gamma=2.5):
        super().__init__(camera, ['preprocess'], stride)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))



    async def process_frame(self, frame, frame_ind):
        '''Perform update on frame, carrying out algorithm'''
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equ = self.clahe.apply(frame)
        #return equ[:,:, np.newaxis]
        return cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)

    async def decorate_frame(self, frame, name):
        # go to BW but don't remove channel
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equ = self.clahe.apply(frame)
        equ = cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)
        return equ


class BackgroundProcessor(Processor):
    '''Substracts and computes background'''
    def __init__(self, camera):
        super().__init__(camera, ['background-removal'], 1)
        self.avg_background = None
        self.count = 0
        self.paused = False

    def get_background(self):
        if self.avg_background is None:
            return None
        result = self.avg_background // self.count
        result = result.astype(np.uint8)
        result = cv2.blur(result, (5,5))
        return result

    def pause(self):
        self.paused = True
    def play(self):
        self.paused = False

    async def process_frame(self, frame, frame_ind):
        '''Perform update on frame, carrying out algorithm'''

        if self.avg_background is None:
            self.avg_background = np.empty(frame.shape, dtype=np.uint32)
            self.avg_background[:] = 0

        if not self.paused:
            self.avg_background += frame
            self.count += 1
        return frame


    async def decorate_frame(self, frame, name):
        return frame - self.get_background()

class TrackerProcessor(Processor):

    @property
    def objects(self):
        '''Objects should have a dictionary with center, bbbox, name, and id'''
        return self._tracking


    def __init__(self, camera, detector_stride, delete_threshold_period=3, stride=1):
        super().__init__(camera, ['track'], stride)
        self._tracking = []
        self.labels = {}
        # need to keep our own ticks because
        # we don't know frame index when track() is called
        self.ticks = 0
        if detector_stride > 0:
            self.ticks_per_obs = detector_stride * delete_threshold_period / self.stride

    async def process_frame(self, frame, frame_ind):
        self.ticks += 1
        delete = []
        for i,t in enumerate(self._tracking):
            status,bbox = t['tracker'].update(frame)
            t['observed'] -= 1
            #update polygon
            if(status):
                t['delta'][0] = bbox[0] - t['init'][0]
                t['delta'][1] = bbox[1] - t['init'][1]
                t['bbox'] = bbox
                t['center_scaled'] = rect_scaled_center(bbox, frame)


            # check obs counts
            if t['observed'] < 0:
                delete.append(i)
        offset = 0
        delete.sort()
        for i in delete:
            del self._tracking[i - offset]
            offset += 1

        return frame

    async def decorate_frame(self, frame, name):
        if name != 'track':
            return frame
        for i,t in enumerate(self._tracking):
            if(t['observed'] < 3):
                continue
            bbox = t['bbox']
            draw_rectangle(frame, bbox, (0,0,255), 1)

            #now draw polygon
            cv2.polylines(frame,[t['poly'] + t['delta']], True, (0,0,255), 3)

            #put note about it
            cv2.putText(frame,
                        '{}: {}'.format(t['name'], t['observed']),
                        (0, 60 * (i+ 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))

        return frame

    def track(self, frame, bbox, poly, label):

        if label in self.labels:
            self.labels[label] += 1
        else:
            self.labels[label] = 1

        #we need to make sure we don't have an existing object here
        for t in self._tracking:
            intersection = intersecting(bbox, t['bbox'])
            if intersection is not None and intersection > 0.25:
                if label != t['label']:
                    #reclassification
                    self.labels[label] -= 1
                    t['name'] = '{}-{}'.format(label, self.labels[label] - 1)
                    t['label'] = label
                # found existing one
                # add to count
                t['observed'] = self.ticks_per_obs
                #update polygon and bounding box and very different
                if intersection < 0.50:
                    t['poly'] = poly
                    t['init'] = bbox
                    t['delta'] = np.int32([0,0])
                    t['center_scaled'] = rect_scaled_center(bbox, frame)
                    t['tracker'] = cv2.TrackerMedianFlow_create()
                    t['tracker'].init(frame, bbox)
                return


        name = '{}-{}'.format(label, self.labels[label] - 1)

        tracker = cv2.TrackerMedianFlow_create()
        status = tracker.init(frame, bbox)

        if not status:
            print('Failed to initialize tracker')
            return False

        track_obj = {'name': name,
                     'tracker': tracker,
                     'label': label,
                     'poly': poly,
                     'init': bbox,
                     'center_scaled': rect_scaled_center(bbox, frame),
                     'bbox': bbox,
                     'observed': self.ticks_per_obs,
                     'start': self.ticks,
                     'delta': np.int32([0,0]),
                     'id': object_id()}
        self._tracking.append(track_obj)
        return True

class SegmentProcessor(Processor):
    def __init__(self, camera, background, stride, max_segments, max_rectangle=0.25):
        '''Pass stride = -1 to only process on request'''
        super().__init__(camera, [

                                  'background-subtract',
                                  'background-thresh',
                                  'background-erode',
                                  'background-open',
                                  'background',
                                  'distance',
                                  'boxes',
                                  'watershed'
                                  ], max(1, stride))
        self.rect_iter = range(0)
        self.background = background
        self.max_segments = max_segments
        self.max_rectangle = max_rectangle
        self.own_process = (stride != -1)

    async def process_frame(self, frame, frame_ind):
        '''we only process on request'''
        if self.own_process:
            return self._process_frame(frame, frame_ind)
        return frame

    def _process_frame(self, frame, frame_ind):
        bg = self._filter_background(frame)
        dist_transform = self._filter_distance(bg)
        self.rect_iter = self._filter_contours(dist_transform)
        return frame

    def segments(self, frame = None):
        if frame is not None:
            self._process_frame(frame, 0)
        yield from self.rect_iter

    def _filter_background(self, frame, name = ''):
        #img = cv2.pyrMeanShiftFiltering(frame, 21, 51)

        img = frame.copy()
        if(img.shape == self.background.shape):
            img -= self.background

        if name == 'background-subtract':
            return img
        img = cv2.blur(img, (6,6))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, bg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if np.mean(bg) > 255 // 2:
            bg = 255 - bg
        #bg = th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                cv2.THRESH_BINARY_INV,11,2)
        if name == 'background-thresh':
            return bg
        # noise removal
        kernel = np.ones((3,3),np.uint8)
        bg = cv2.erode(bg, kernel, iterations = 3)
        if name == 'background-erode':
            return bg
        bg = cv2.morphologyEx(bg,cv2.MORPH_OPEN,kernel, iterations = 3)
        if name == 'background-open':
            return bg

        return bg

    def _filter_distance(self, frame):
        dist_transform = cv2.distanceTransform(frame, cv2.DIST_L2,5)
        dist_transform = cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)

        #create distance tranform contours
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

    def _filter_contours(self, frame, return_contour=False):
        _, contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        def sort_key(c):
            '''The area of bounding rectangle'''
            rect = cv2.boundingRect(c)
            return rect[2] * rect[3]
        contours.sort(key = sort_key, reverse=True)
        rects = [cv2.boundingRect(c) for c in contours]
        segments = 0

        for c,r in zip(contours, rects):
            #flip around our rectangle
            # exempt small or large rectangles
            if(r[2] * r[3] < 250 or \
                r[2] * r[3] / frame.shape[0] / frame.shape[1] > self.max_rectangle ):
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
        bg = self._filter_background(frame, name)
        if name.find('background') != -1:
            return bg

        dist_transform = self._filter_distance(bg)
        if name == 'distance':
            return dist_transform

        if name == 'boxes':
            for rect in self._filter_contours(dist_transform):
                draw_rectangle(frame, rect, (255, 255, 0), 1)
        if name == 'watershed':
            markers = self._filter_ws_markers(dist_transform)
            ws_markers = cv2.watershed(frame, markers)
            frame[ws_markers == -1] = (255, 0, 0)
        return frame


class TrainingProcessor(Processor):

    @property
    def objects(self):
        '''Objects should have a dictionary with center, bbbox, name, and id'''
        return self._objects

    ''' This will segment an ROI from the frame and you can label it'''
    def __init__(self, camera, background, img_db, descriptor, max_segments=3):
        super().__init__(camera, ['training'], 1)
        self.segmenter = SegmentProcessor(camera, background, -1, max_segments)
        self.img_db = img_db
        self.rect_index = 0
        self.poly_index = 0
        self.poly_len = 0
        self.rect_len = 0
        self.rect = (0,0,100,100)
        self.segments = []
        self.polys = []
        self.poly = np.array([[0,0], [0,0]])
        self.descriptor = descriptor
        self._objects = []

    def close(self):
        super().close()
        self.segmenter.close()

    def set_descriptor(self, desc):
        self.descriptor = desc

    async def process_frame(self, frame, frame_ind):
        self.segments = list(self.segmenter.segments(frame))
        self.rect_len = len(self.segments)
        if self.rect_index >= 0 and self.rect_index < len(self.segments):
            self.rect = stretch_rectangle(self.segments[self.rect_index], frame)

        # index 1 is poly
        self.polys = [x[0] for x in self.segmenter.polygon(frame, self.rect)]
        self.poly_len = len(self.polys)
        if self.poly_index >= 0 and self.poly_index < len(self.polys):
            self.poly = self.polys[self.poly_index]

        return frame

    async def decorate_frame(self, frame, name):

        if name == 'training':


            kp, _ = keypoints_view(self.descriptor, frame, self.rect)
            cv2.drawKeypoints(frame, kp, frame, color=(32,32,32), flags=0)

            for r in self.segments:
                draw_rectangle(frame, r, (60, 60, 60), 1)
            draw_rectangle(frame, self.rect, (255, 255, 0), 3)

            frame_view = rect_view(frame, self.rect)
            for p in self.polys:
                cv2.polylines(frame_view, [p], True, (60, 60, 60), 1)
            cv2.polylines(frame_view, [self.poly], True, (0, 0, 255), 3)

        return frame

    def capture(self, frame, label):
        '''Capture and store the current image'''
        img = rect_view(frame, self.rect)
        # process it
        kp = self.descriptor.detect(img, None)
        if(len(kp) < 4):
            return False
        processed = img.copy()
        cv2.drawKeypoints(processed, kp, processed, color=(32,32,32), flags=0)
        cv2.polylines(processed, [self.poly], True, (0,0,255), 3)
        self.img_db.store_img(img, label, self.poly, kp, processed)

        #create obj
        self._objects = [{
            'bbox': self.rect,
            'center_scaled': rect_scaled_center(self.rect, frame),
            'label': label,
            'id': object_id()
        }]

        return True


class DetectionProcessor(Processor):
    '''Detects query images in frame. Uses async to spread out computation. Cannot handle replicas of an object in frame'''
    def __init__(self, camera, background, img_db, descriptor, stride=5,
                 threshold=0.8, template_size=256, min_match=6,
                 weights=[3, -1, -1, -10, 5], max_segments=10,
                 track=True):

        #we have a specific order required
        #set-up our tracker
        # give estimate of our stride
        if track:
            self.tracker = TrackerProcessor(camera, stride * 2 * len(img_db))
        else:
            self.tracker = None
        #then our segmenter
        self.segmenter = SegmentProcessor(camera, background, -1, max_segments)
        #then us
        super().__init__(camera, ['keypoints', 'identify'], stride)


        self._ready = True
        self.features = {}
        self.threshold = threshold
        self.min_match = min_match
        self.weights = weights
        self.stretch_boxes=1.5
        self.track = track
        self.templates = img_db


        # Initiate descriptors
        self.desc = descriptor
        #set-up our matcher
        self.matcher = cv2.BFMatcher()

        #create color gradient
        N = len(img_db)
        cm = plt.cm.get_cmap('Dark2')
        for i,t in enumerate(self.templates):
            rgba = cm(i / N)
            rgba = [int(x * 255) for x in rgba]
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
            asyncio.ensure_future(self._identify_features(frame.copy()))
        return frame

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
                points = f['poly']
                color = f['color']
                kp = f['kp']
                kpcolor = f['kpcolor']
                for p,c in zip(kp, kpcolor):
                    cv2.circle(frame, tuple(p), 6, color, thickness=-1)

                #draw polygon
                cv2.polylines(frame,[points],True, color, 3, cv2.LINE_AA)
                #get bottom of polygon
                cv2.putText(frame, '{} ({:.2})'.format(n, f['score']),
                            (f['rect'][0], f['rect'][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color)

        return frame

    async def _identify_features(self, frame):
        self._ready = False
        #make new features object
        features = {}

        found_feature = False
        for rect in self.segmenter.segments(frame):
            kp, des = keypoints_view(self.desc, frame, rect)
            if(des is not None and len(des) > 3):
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
                template = t.img
                name = t.label
                descriptors = t.keypoints, t.features


                matches = self.matcher.knnMatch(descriptors[1], des, k=2)
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
                    dst_bbox = cv2.boundingRect(dst_poly)

                    # check if the polygon is actually good
                    area = max(0.01, cv2.contourArea(dst_poly))
                    perimter = max(0.01, cv2.arcLength(dst_poly, True))
                    score = len(good) / len(des) * self.weights[0] + \
                            perimter / area * self.weights[1] + \
                            (dst_bbox[2] / bounds[2] - 1 + dst_bbox[3] /  bounds[3] - 1) * self.weights[2] + \
                            (dst_bbox[2] * dst_bbox[3] < 5) * self.weights[3] + \
                            self.weights[4]
                    if score > 0:
                        cm = plt.cm.get_cmap()
                        features[name] = { 'color': t.color, 'poly': np.int32(dst_poly),
                            'kp': np.int32([kp[m.trainIdx].pt for m in good]).reshape(-1,2),
                            'kpcolor': [cm(x.distance / good[-1].distance) for x in good],
                            'score': score, 'rect': bounds}
                        # register it with our tracker
                        if self.track:
                            self.tracker.track(frame, bounds, np.int32(dst_poly), name)
            except cv2.error:
                #not enough points
                await asyncio.sleep(0)
                continue

            #cede control
            await asyncio.sleep(0)
        return features
