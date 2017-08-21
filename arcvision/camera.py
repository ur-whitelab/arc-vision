#!/usr/bin/python3
'''web camera related class/functions'''

import asyncio
import cv2
import copy

#The async is so that the program can yield control to other asynchronous tasks

class Camera:
    '''Class for managing processing video frames'''
    def __init__(self, video_file=-1, frame_buffer=10):

        if video_file == '':
            video_file = 0
        #check if what is passed corresponds to an integer
        try:
            int_video_file = int(video_file)
            video_file = int_video_file
        except ValueError:
            pass
        print('Camera using file {}'.format(video_file))
        self.video_file = video_file
        self.sem = asyncio.Semaphore(frame_buffer)
        self.frame_processors = []
        self.frame = None
        self.decorated_frame = None
        self.decorate_index = 0

        self.cap = cv2.VideoCapture(self.video_file)

    def add_frame_processor(self, p):
        '''Add a frame processor object'''
        assert hasattr(p, 'process_frame')
        assert hasattr(p, 'decorate_frame')
        assert hasattr(p, 'stride')
        self.frame_processors.append(p)

    def remove_frame_fxn(self, p):
        '''Remove a frame processor object from being updated'''
        i = self.frame_processors.index(p)
        del self.frame_processors[i]


    async def _process_frame(self, frame_ind):
        '''Process the frames. We only update the decorated frame when necessary'''

        update_decorated = False
        if self.decorate_index > 0:
            #check if the requested decorated frame will be updated
            update_decorated = self.frame_processors[self.decorate_index - 1].stride % frame_ind == 0 #off by one so 0 can indicate no processing
        if update_decorated or self.decorate_index == 0:
            self.decorated_frame = copy.copy(self.frame)

        for i,p in enumerate(self.frame_processors):
            if frame_ind % p.stride == 0:
                #process frame
                p.process_frame(self.frame)
            #if we are updating the decorated frame, then we must
            if(i < self.decorate_index and update_decorated):
                p.decorate_frame(self.decorated_frame)

        self.sem.release()

    async def update(self):
        '''Process an update from the camera feed'''
        frame_ind = 1
        if self.cap.isOpened():
            await self.sem.acquire()
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                task = asyncio.ensure_future(self._process_frame(frame_ind))
                frame_ind += 1
                await asyncio.sleep(0)
                await asyncio.gather(task)
                return True
        return False

    def get_frame(self):
        return self.frame

    def save_frame(self,frame,file_location):
        cv2.imwrite(file_location,frame)

    def get_decorated_frame(self, index=-1):
        '''Retrive the decorated frame and specify which one you want. Use negative to indicate last'''
        if(index < 0):
            index = len(self.frame_processors) - 1
        self.decorate_index = index
        return self.decorated_frame


