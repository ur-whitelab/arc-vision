#!/usr/bin/python3
'''web camera related class/functions'''

import asyncio
import cv2
import hashlib
import copy

class Camera:
    '''Class for managing processing video frames'''
    def __init__(self, video_file=-1, frame_buffer=10):
        self.video_file = video_file
        self.sem = asyncio.Semaphore(frame_buffer)
        self.frame_fxns = []
        self.frame_strides = []
        self.frame = None

    def add_frame_fxn(self, fxn, stride=1):
        '''Add a function which will be called each stride frames'''
        self.frame_fxns.append(fxn)
        self.frame_strides.append(stride)
    def remove_frame_fxn(self, fxn):
        i = self.frame_fxns.index(fxn)
        del self.frame_fxns[i]
        del self.frame_strides[i]

    async def _process_frame(self, frame, frame_ind):
        for s,f in zip(self.frame_strides, self.frame_fxns):
            if frame_ind % s == 0:
                await f(frame)
        self.sem.release()

    async def start(self):
        '''This starts processing the camera feed'''
        cap = cv2.VideoCapture(self.video_file)
        tasks = list()
        frame_ind = 0
        while cap.isOpened():
            await self.sem.acquire()
            ret, frame = cap.read()
            if not ret:
                break
            self.frame = frame
            tasks.append(asyncio.ensure_future(self._process_frame(frame, frame_ind)))
            frame_ind += 1
            await asyncio.sleep(0)
        await asyncio.gather(tasks)

    def get_frame(self):
        return self.frame

