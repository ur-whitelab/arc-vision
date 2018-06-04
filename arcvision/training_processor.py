import numpy as np
import cv2
from .utils import *

from .processor import Processor
from .segment_processor import SegmentProcessor

class TrainingProcessor(Processor):

    @property
    def objects(self):
        '''Objects should have a dictionary with center, bbrect, name, and id'''
        return self._objects

    ''' This will segment an ROI from the frame and you can label it'''
    def __init__(self, camera, img_db, descriptor, background = None, max_segments=3):
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
            self.rect = self.segments[self.rect_index]#stretch_rectangle(self.segments[self.rect_index], frame)

        # index 1 is poly
        self.polys = [x[0] for x in self.segmenter.polygon(frame, self.rect)]
        self.poly_len = len(self.polys)
        if self.poly_index >= 0 and self.poly_index < len(self.polys):
            self.poly = self.polys[self.poly_index]

        return

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

        return frame#cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
            'brect': self.rect,
            'poly': self.poly,
            'center_scaled': poly_scaled_center(self.poly, frame) if cv2.contourArea(self.poly) < rect_area(self.rect) else rect_scaled_center(self.rect, frame),
            'label': label,
            'id': object_id()
        }]

        return True