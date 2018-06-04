import numpy as np
from numpy import linalg
import os, pickle, pathlib
from .utils import *

from .processor import Processor
from .segment_processor import SegmentProcessor


class SpatialCalibrationProcessor(Processor):
    '''This will find a perspective transform that goes from our coordinate system
       to the projector coordinate system. Convergence in done by using point guess in next round with
       previous round estimate'''

    ''' Const for the serialization file name '''
    PICKLE_FILE = pathlib.Path('.') / 'calibrationdata' / 'spatialCalibrationData.p'

    def __init__(self, camera, background=None, channel=1, stride=1, N=16, delay=10, stay=20, readAtInit = True, segmenter = None):
        #stay should be bigger than delay
        #stay is how long the calibration dot stays in one place (?)
        #delay is how long we wait before reading its position
        if segmenter is None:
            self.segmenter = SegmentProcessor(camera, background, -1, 4, max_rectangle=0.25, channel=channel, name='Spatial')
        else:
            self.segmenter = segmenter
        super().__init__(camera, ['calibration', 'transform'], stride)
        self.calibration_points = np.random.random( (N, 2)) * 0.8 + 0.1

        o = {}
        o['id'] = object_id()
        o['center_scaled'] = None
        o['label'] = 'calibration-point'
        self._objects = [o]
        self.index = 0
        self.delay = delay
        self.stay = stay
        self.N = N
        self.first = True
        self.channel = channel
        self.readAtReset = readAtInit
        self.frameWidth = camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frameHeight = camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.reset()


    @property
    def background(self):
        return None

    @background.setter
    def background(self, background):
        self.segmenter.background = background


    @property
    def transform(self):
        return self._best_scaled_transform

    @property
    def inv_transform(self):
        return linalg.inv(self._best_scaled_transform)

    def close(self):
        super().close()
        self.segmenter.close()

    def play(self):
        self.calibrate = True

    def _read_calibration(self, filepath):
        if(os.path.exists(filepath)):
            # check if the currently read file has an entry for this resolution
            allData = pickle.load(open(filepath, 'rb'))
            res_string = '{}x{}'.format(self.frameWidth, self.frameHeight)
            if (res_string in allData):
                print(f'Reading homography from {filepath}')
                data = allData[res_string]
                self.first = False
                self._transform = data['transform']
                self._scaled_transform = data['scaled_transform']
                self._best_scaled_transform = data['best_scaled_transform']
                self._best_list = data['best_list']
                self._best_inv_list = data['best_inv_list']
                self.fit = data['fit']
                self._best_fit = 0.01
                self.calibrate = False
                self.first = False
                self.initial_fit = self.fit
                return True
        return False

    def _write_calibration(self, filepath):
        if (os.path.exists(filepath)):
                # read the existing data file to update
                data = pickle.load(open(filepath, 'rb'))
        if (os.path.exists(filepath)):
                  # read the existing data file to update
            data = pickle.load(open(filepath, 'rb'))
        else:
            # start fresh
            data = {}
        # create a sub-dict for this resolution
        subData = {}
        subData['transform'] = self._transform
        subData['scaled_transform']=self._scaled_transform
        subData['best_scaled_transform'] = self._best_scaled_transform
        subData['best_list'] = self._best_list
        subData['best_inv_list'] = self._best_inv_list
        subData['fit'] = self.fit
        subData['width'] = self.frameWidth
        subData['height'] = self.frameHeight

        # add it to the existing dictionary, then write the updated data out
        data['{}x{}'.format(self.frameWidth, self.frameHeight)] = subData
        print(f'Writing calibration to {filepath}')
        #make sure directory is ready
        try:
            os.makedirs(filepath.parent)
        except FileExistsError:
            #OK if it exists
            pass
        pickle.dump(data, open(filepath, 'wb'))

    def pause(self):
        # only write good fits/better than the previously calculated one
        if (self.fit < .001 and self.fit < self.initial_fit):
            self._write_calibration(SpatialCalibrationProcessor.PICKLE_FILE)

        self.calibrate = False



    def reset(self):
        self.points = np.zeros( (self.N, 2) )
        self.counts = np.zeros( (self.N, 1) )
        #try to read the pickle file
        if not self.readAtReset or not self._read_calibration(SpatialCalibrationProcessor.PICKLE_FILE):
            #didn't work, set to defaults
            self._transform = np.identity(3)
            self._scaled_transform = np.identity(3)
            self._best_scaled_transform = np.identity(3)
            self._best_fit = 0.01 #reasonable amount, anything less shouldn't be used
            self._best_list = np.array([1., 0., 0., 0., 1., 0., 0., 0. ,1.])
            self._best_inv_list = np.array([1., 0., 0., 0., 1., 0., 0., 0., 1.])
            self.fit = 100
            self.initial_fit = self.fit
            self.calibrate = False


    async def process_frame(self, frame, frame_ind):
        if self.calibrate:
            self._calibrate(frame, frame_ind)
        return

    def _calibrate(self, frame, frame_ind):
        if frame_ind % (self.stay + self.delay) > self.delay:
            for seg in self.segmenter.segments(frame):
                #if(rect_color_channel(frame, seg) == self.channel):
                p = rect_scaled_center(seg, frame)
                self.points[self.index, :] = self.points[self.index, :] * self.counts[self.index] / (self.counts[self.index] + 1) + p / (self.counts[self.index] + 1)
                self.counts[self.index] += 1
                break


        if frame_ind % (self.stay + self.delay) == 0:
            # update homography estimate
            if self.index == self.N - 1:
                print('updating homography...fit = {}'.format(self.fit))
                self._update_homography(frame)
                if (self.fit < .001 and self.fit < self.initial_fit):
                    self._write_calibration(SpatialCalibrationProcessor.PICKLE_FILE)
                self.calibration_points = np.random.random( (self.N, 2)) * 0.8 + 0.1
                #seed next round with fit, weighted by how well the homography fit
                self.points[:] = cv2.perspectiveTransform(self.calibration_points.reshape(-1,1,2), linalg.inv(self._transform)).reshape(-1,2)
                self.counts[:] = max(0, (0.01 - self.fit) * 10)
            self.index += 1
            self.index %= self.N

    def warp_img(self, img):
        #return cv2.warpPerspective(img, self._best_scaled_transform, (img.shape[1], img.shape[0]))
        for i in range(img.shape[2]):
            img[:,:,i] = cv2.warpPerspective(img[:,:,i],
                                             self._best_scaled_transform,
                                             img.shape[1::-1])
        return img

    def warp_point(self, point):
        w = point[0]* self._best_list[6] + point[1] * self._best_list[7] + self._best_list[8]
        x = point[0] * self._best_list[0] + point[1] * self._best_list[1] + self._best_list[2]
        y = point[0] * self._best_list[3] + point[1] * self._best_list[4] + self._best_list[5]
        if (w != 0):
            point[0] = x/w
            point[1] = y/w
        else:
            point[0] = 0
            point[1] = 0
        return point

    def unwarp_point(self, point):
        w = point[0]* self._best_inv_list[6] + point[1] * self._best_inv_list[7] + self._best_inv_list[8]

        x = point[0] * self._best_inv_list[0] + point[1] * self._best_inv_list[1] + self._best_inv_list[2]
        y = point[0] * self._best_inv_list[3] + point[1] * self._best_inv_list[4] + self._best_inv_list[5]
        if (w != 0):
            point[0] = x/w
            point[1] = y/w
        else:
            point[0] = 0
            point[1] = 0
        return point

    def _update_homography(self, frame):
        if(np.sum(self.counts > 0) < 5):
            return
        t, mask = cv2.findHomography((self.points[self.counts[:,0] > 0, :]).reshape(-1, 1, 2),
                                self.calibration_points[self.counts[:,0] > 0, :].reshape(-1, 1, 2),
                                0)
        p = cv2.perspectiveTransform((self.points).reshape(-1, 1, 2), self._transform).reshape(-1, 2)
        ts, _ = cv2.findHomography(self._unscale(self.points[self.counts[:,0] > 0, :], frame.shape).reshape(-1, 1, 2),
                    self._unscale(self.calibration_points[self.counts[:,0] > 0,:], frame.shape).reshape(-1, 1, 2),
                    0)
        if t is None:
            print('homography failed')
        else:
            if(self.first):
                self._transform = t
                self._scaled_transform = ts
                self.first = False
            else:
                self._transform = self._transform * 0.6 + t * 0.4
                self._scaled_transform = self._scaled_transform * 0.6 + ts * 0.4

        # get fit relative to identity
        self.fit = linalg.norm(self.calibration_points.reshape(-1, 1, 2) - cv2.perspectiveTransform((self.points).reshape(-1, 1, 2), self._transform)) / self.N
        if self.fit < self._best_fit:
            self._best_scaled_transform = self._scaled_transform
            self._best_fit = self.fit
            self._best_list = (self._transform).flatten()
            self._best_inv_list = (linalg.inv(self._transform)).flatten()

    def _unscale(self, array, shape):
        return (array * [shape[1], shape[0]]).astype(np.int32)

    async def decorate_frame(self, frame, name):
        if name == 'transform':
            self.warp_img(frame)
        if name == 'calibration' or name == 'transform':
            for i in range(self.N):
                p = np.copy(self.calibration_points[i, :])

                c = self.unwarp_point(p)
                c = self._unscale(c, frame.shape)

                #BGR

                # points represents the found location of the calibration circle (printed in red)
                cv2.circle(frame,
                            tuple(self._unscale(self.points[i],
                                frame.shape)), 10, (0,0,255), -1)

                # c is the ground-truth location of the calibration circle, unwarped to be in CV space
                # printed in blue. these should be close to the red points if the fit is good
                cv2.circle(frame,
                            tuple(c.astype(np.int)), 10, (255,0,0), -1)

                # calibration points
                cv2.circle(frame,
                            tuple(self._unscale(self.calibration_points[i, :],
                                frame.shape)), 10, (0,255, 255), -1)
                # draw a purple line between the corresponding red and blue dot to track distance
                cv2.line(frame, tuple(self._unscale(self.points[i],
                                frame.shape)), tuple(c.astype(np.int)), (255,0,255))
            # draw rectangle of the transform
            p_rect = np.array( [[0, 0], [0, 1], [1, 1], [1, 0]], np.float).reshape(-1, 1, 2)
            c_rect = cv2.perspectiveTransform(p_rect, (self._transform))
            c_rect = self._unscale(c_rect, frame.shape)
            cv2.polylines(frame, [c_rect], True, (0, 255, 125), 4)
            cv2.putText(frame,
                'Homography Fit: {}'.format(self.fit),
                (100, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,124,255))
        return  frame#cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @property
    def objects(self):
        if self.calibrate:
            self._objects[0]['center_scaled'] = self.calibration_points[self.index]
            return self._objects
        return []