#!/usr/bin/python3
'''web camera related class/functions'''

import asyncio
import cv2

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
        self.frame_fxns = []
        self.frame_strides = []
        self.frame = None

        self.cap = cv2.VideoCapture(self.video_file)

    def add_frame_fxn(self, fxn, stride=1):
        '''Add a function which will be called each stride frames'''
        self.frame_fxns.append(fxn)
        self.frame_strides.append(stride)

    def remove_frame_fxn(self, fxn):
        '''Remove a function from being updated'''
        i = self.frame_fxns.index(fxn)
        del self.frame_fxns[i]
        del self.frame_strides[i]

    async def _process_frame(self, frame, frame_ind):
        for s,f in zip(self.frame_strides, self.frame_fxns):
            if frame_ind % s == 0:
                f(frame)
        self.sem.release()

    async def update(self):
        '''Process an update from the camera feed'''
        frame_ind = 0
        if self.cap.isOpened():
            await self.sem.acquire()
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                task = asyncio.ensure_future(self._process_frame(frame, frame_ind))
                frame_ind += 1
                await asyncio.sleep(0)
                await asyncio.gather(task)
                return True
        return False

    def get_frame(self):
        return self.frame

    def save_frame(self,frame,file_location):
        cv2.imwrite(file_location,frame)

