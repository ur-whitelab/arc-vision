import numpy as np
import cv2
from .utils import *

# from processor import Processor
from .processor import Processor

class BackgroundProcessor(Processor):
    '''Substracts and computes background'''
    def __init__(self, camera, background = None):
        super().__init__(camera, ['bg-view', 'bg-diff-blur'], 1)
        self.reset()
        self.pause()
        self._background = background
        self._blank = None

    @property
    def background(self):
        return self._background

    def pause(self):
        self.paused = True
    def play(self):
        self.paused = False
    def reset(self):
        self.count = 0
        self.avg_background = None
        self.paused = False

    async def process_frame(self, frame, frame_ind):
        '''Perform update on frame, carrying out algorithm'''
        #cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.avg_background is None:
            self.avg_background = np.empty(frame.shape, dtype=np.uint32)
            self.avg_background[:] = 0
            self._background = frame.copy()
            self._blank = np.zeros((frame.shape))

        if not self.paused:
            self.avg_background += frame
            self.count += 1
            self._background = self.avg_background // max(1, self.count)
            self._background = self._background.astype(np.uint8)
            #self._background = cv2.blur(self._background, (5,5))
        return


    async def decorate_frame(self, frame, name):
        if name == 'bg-view':
            return self.background
        if name =='bg-diff-blur':
            return diff_blur(self.background, frame)
        return frame