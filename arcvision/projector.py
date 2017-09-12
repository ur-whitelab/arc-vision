import numpy as np
import cv2
import asyncio

from .processor import Processor

class Projector(Processor):
    '''This will handle retrieving and caching the frame displayed by the proejctor at each frame'''
    def __init__(self, camera, projector_socket):
        super().__init__(camera, ['frame', 'transformed'], 1)
        self.sock = projector_socket
        self._transform = np.identity(3)


    async def process_frame(self, frame, frame_ind):
        try:
            await asyncio.wait_for(self.sock.send('{}-{}'.format(frame.shape[1], frame.shape[0]).encode()), timeout=0.1)
            response = await asyncio.wait_for(self.sock.recv(), timeout=0.1)
            jpg = np.fromstring(response, np.uint8)
            img = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
            self._frame = img
            self._transformed_frame = img.copy()
            for i in range(frame.shape[2]):
                self._transformed_frame[:,:,i] = cv2.warpPerspective(self._transformed_frame[:,:,i],
                                                                     self._transform,
                                                                     frame.shape[1::-1])
        except asyncio.TimeoutError:
            print('Unable to connect to projector...')
        return frame

    async def decorate_frame(self, frame, name):
        if name =='frame':
            return self._frame
        elif name == 'transformed':
            return self.frame
        return frame

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def tranform(self, value):
        self._transform = value

    @property
    def frame(self):
        return self._transformed_frame
