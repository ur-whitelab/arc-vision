#!/usr/bin/python3
'''web camera related class/functions'''

import asyncio
import cv2
import copy
import numpy as np
import sys
import time
import atexit

#The async is so that the program can yield control to other asynchronous tasks

class Camera:
    '''Class for managing processing video frames'''
    def __init__(self, video_file=-1, frame_buffer=1, output=None):

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
        self.decorate_name = 'raw'
        self.frame_ind = 1
        self.stream_names = {'Base': ['raw']}
        self.paused = False
        self.cap = cv2.VideoCapture(self.video_file)
        self.output = output
        atexit.register(self.close)



        # try turning off autofocus
        # could be mp4 file, so catch error
        try:
            #self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)#1280
            #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)#720
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)#800
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)#450
            self.cap.set(cv2.CAP_PROP_FPS, 60)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.cap.set(cv2.CAP_PROP_HUE, 1.0)
            self.cap.set(cv2.CAP_PROP_SATURATION, 225.0)
            self.cap.set(cv2.CAP_PROP_GAIN, 2.0)

        except cv2.error:
            pass

    def close(self):
        if self.output is not None:
            self.output.close()
        self.cap.close()

    def add_frame_processor(self, p):
        '''Add a frame processor object'''
        assert hasattr(p, 'process_frame')
        assert hasattr(p, 'decorate_frame')
        self.frame_processors.append(p)
        self.stream_names[p.name] = p.streams

    def remove_frame_processor(self, p):
        '''Remove a frame processor object from being updated'''
        self.frame_processors.remove(p)
        del self.stream_names[p.name]

        # I'm a simple man
        self.decorate_index = min(len(self.frame_processors), self.decorate_index)


    async def _process_frame(self, frame, frame_ind):
        '''Process the frames. We only update the decorated frame when necessary'''

        # we will update if we are at raw always
        decorated_frame = None
        update_decorated = self.decorate_index == 0
        if self.decorate_index > 0:
            #check if the requested decorated frame will be updated
            update_decorated =  frame_ind % self.frame_processors[self.decorate_index - 1].stride == 0 #off by one so 0 can indicate no processing
        if update_decorated or self.decorate_index == 0:
            decorated_frame = frame.copy()

        start_dims = self.frame.shape
        for i,p in enumerate(self.frame_processors):
            if frame_ind % p.stride == 0:
                #process frame
                #startTime = time.time()
                await p.process_frame(frame, frame_ind)
                #endTime = time.time()
                #elp = endTime - startTime
                #if (elp != 0.0):
                #    print("Elapsed time using processor {} is {} seconds, frame index is {}".format(p.name, endTime - startTime, frame_ind))

                assert frame is not None, \
                    'Processor {} returned None on Process Frame {}'.format(type(p).__name__, self.frame_ind)

                assert len(self.frame.shape) == len(start_dims), \
                    'Processor {} modified frame channel from {} to {}'.format(type(p), start_dims, self.frame.shape)
            #if we are updating the decorated frame, then we must
            if(i < self.decorate_index and update_decorated):
                decorated_frame = await p.decorate_frame(decorated_frame, self.decorate_name)

                # lots of steps, if we lose color channel add it back
                if decorated_frame is not None and len(decorated_frame.shape) == 2:
                    decorated_frame = cv2.cvtColor(decorated_frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)

                if decorated_frame is None:
                    'Processer {} returned None on Decorate Frame {}'.format(type(p).__name__, self.frame_ind)
                    decorated_frame = frame.copy()
        if self.output is not None:
            if type(self.output) == str:
                print('Beginning to write to {}'.format(self.output))
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.output = cv2.VideoWriter(self.output, fourcc, 30.0, (self.frame.shape[1],self.frame.shape[0]))
            if decorated_frame is not None:
                self.output.write(decorated_frame)
            else:
                self.output.write(self.decorated_frame)
        if update_decorated:
            self.decorated_frame = decorated_frame
        self.sem.release()

    def pause(self):
        if self.paused:
            return

        self.paused = True

        # try to read. If we fail, we re-use the last frame
        # which could have processing artifacts. Best we can do though
        ret, frame = self.cap.read()
        if frame is not None:
            self.raw_frame = frame
            self.frame_ind += 1
        else:
            self.raw_frame = self.frame

    def play(self):
        self.paused = False
    async def update(self):
        '''Process an update from the camera feed'''
        if self.cap.isOpened():
            await self.sem.acquire()
            # check this, for if we have a looping video
            ret, frame = await self._cap_frame()
            if ret and frame is not None:
                # normal update
                self.frame = frame
                task = asyncio.ensure_future(self._process_frame(frame, self.frame_ind))
                await asyncio.sleep(0)
                await asyncio.gather(task)
                return True
        return False

    def _flush_buffers(self, N=5):
        for i in range(N):
            self.cap.grab()

    async def _cap_frame(self):
        if self.frame_ind - 1 == self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
            print('Completed video, looping again')
            self.frame_ind = 1
            self.cap = cv2.VideoCapture(self.video_file)
        if not self.paused:
            ret, frame = self.cap.read()
            self.frame_ind += 1
        else:
            ret, frame = True, self.raw_frame.copy()
        return ret, frame

    def get_frame(self):
        return self.frame

    def save_frame(self,frame,file_location):
        cv2.imwrite(file_location,frame)

    def get_decorated_frame(self, name):
        if name == 'raw' or len(self.frame_processors) == 0:
            self.decorate_index = 0
            return cv2.pyrDown(self.decorated_frame)

        for i, p in enumerate(self.frame_processors):
            if name in p.streams:
                break
        self.decorate_index = i + 1
        if not name in p.streams:
            # bad name, so just give last one
            #print("THERE ARE {} STREAMS".format(len(p.streams)))
            name = p.streams[-1]
        self.decorate_name = name
        return self.decorated_frame


