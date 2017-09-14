import numpy as np
import cv2
import asyncio

from .processor import Processor



class Projector(Processor):
    '''This will handle retrieving and caching the frame displayed by the proejctor at each frame'''
    def __init__(self, camera, projector_socket):
        super().__init__(camera, ['frame', 'transformed'], 1, has_consumer=True)
        self.sock = projector_socket
        self._transform = np.identity(3)
        self._transformed_frame = None
        self._frame = None


    async def process_frame(self, frame, frame_ind):

        shape = frame.shape
        try:
            await asyncio.wait_for(self.sock.send('{}-{}'.format(shape[1], shape[0]).encode()), timeout=0.02)
            response = await asyncio.wait_for(self.sock.recv(), timeout=0.02)
            self._queue_work( (response, self._transform, shape) )

        except asyncio.TimeoutError:
            print('Unable to connect to projector...')
        return frame

    async def decorate_frame(self, frame, name):
        if name =='frame':
            return self._frame
        elif name == 'transformed':
            return self.frame
        return frame


    @classmethod
    def _process_work(cls, data):
        response, transform, shape = data
        jpg = np.fromstring(response, np.uint8)
        img = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
        t_img = img.copy()
        for i in range(shape[2]):
                t_img[:,:,i] = cv2.warpPerspective(t_img[:,:,i],
                                                   transform,
                                                   shape[1::-1])
        return img, t_img

    def _process_result(self, result):
        self._frame, self._transformed_frame = result

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def tranform(self, value):
        self._transform = value

    @property
    def frame(self):
        return self._transformed_frame
