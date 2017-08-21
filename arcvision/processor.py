import asyncio
import random
import cv2

class ExampleProcessor:
    '''Class that detects multiple objects given a set of labels and images'''
    def __init__(self, width=500, height=500, stride=1):
        self.corners = []
        self.corners.append((random.randrange(0, width / 4), random.randrange(0, height / 4)))
        self.corners.append((random.randrange(3 * width / 4, width), random.randrange(3 * height / 4, height)))
        self.color = tuple([random.randrange(0,255) for i in range(3)])
        self.stride = stride

    def process_frame(self, frame):
        '''Perform update on frame, carrying out algorithm'''
        #simulate working hard
        asyncio.sleep(0.25)

    def decorate_frame(self, frame):
        '''Draw visuals onto the given frame, without carrying-out update'''
        cv2.rectangle(frame, *self.corners, self.color, 2)

