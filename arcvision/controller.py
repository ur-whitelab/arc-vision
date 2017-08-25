import zmq
import zmq.asyncio
import time
import argparse
import asyncio
import glob
import os
import sys
from .camera import Camera
from .server import start_server
from .calibration import Calibrate
from .processor import *
from .utils import *
import json

from .protobufs.reactors_pb2 import ReactorSystem


zmq.asyncio.install()

class Controller:
    '''Controls flow of reactor program'''
    def __init__(self, zmq_sub_port, zmq_pub_port, cc_hostname):
        self.ctx = zmq.asyncio.Context()

        #subscribe to publishing socket
        zmq_uri = 'tcp://{}:{}'.format(cc_hostname, zmq_sub_port)
        print('Connecting SUB Socket to {}'.format(zmq_uri))
        self.projector_sock = self.ctx.socket(zmq.SUB)
        self.projector_sock.connect(zmq_uri)
        #we only want vision updates
        sub_topic = 'projector-update'
        print('listening for topic: {}'.format(sub_topic))
        self.projector_sock.subscribe(sub_topic.encode())

        #register publishing socket
        zmq_uri = 'tcp://{}:{}'.format(cc_hostname, zmq_pub_port)
        print('Connecting PUB Socket to {}'.format(zmq_uri))
        self.pub_sock = self.ctx.socket(zmq.PUB)
        self.pub_sock.connect(zmq_uri)

        #statistics
        self.frequency = 1
        self.stream_names = []

        #create state
        self.vision_state = ReactorSystem()
        self.vision_state.time = 0

        #settings
        self.settings = {'mode': 'background', 'pause': False, 'descriptor': 'BRISK'}
        self.modes = ['background', 'detection', 'training']
        self.descriptors = ['BRISK', 'AKAZE']
        self.processors = []
        self.background = None

    def get_state_json(self):
        if self.settings['mode'] == 'training':
            self.settings['training_poly_len'] = self.processors[0].poly_len
            self.settings['training_rect_len'] = self.processors[0].rect_len
            self.settings['training_rect_index'] = self.processors[0].rect_index
            self.settings['training_poly_index'] = self.processors[0].poly_index
        return json.dumps(self.__dict__, default=lambda x: '')


    async def handle_start(self, video_filename, server_port, template_dir, crop):
        '''Begin processing webcam and updating state'''

        self.cam = Camera(video_filename)
        self.img_db = ImageDB(template_dir)
        start_server(self.cam, self, server_port)
        print('Started arcvision server')

        # setup initial streams
        if crop is not None:
            CropProcessor(self.cam, crop)
        self.processors = [BackgroundProcessor(self.cam)]

        self.update_settings(self.settings)

        while True:
            sys.stdout.flush()
            await self.update_loop()

    def _reset_processors(self):

        for p in self.processors:
            if p.__class__ == BackgroundProcessor:
                bg = p.get_background()
                if bg is not None:
                    self.background = bg
        [x.close() for x in self.processors]
        self.processors = []

    def _start_detection(self):
        return
        self.processors = [DetectionProcessor(self.cam, self.background, paths, labels)]

    def update_settings(self, settings):
        if 'mode' in settings and settings['mode'] != self.settings['mode']:
            mode = settings['mode']

            if mode == 'detection':
                self._reset_processors()
                self._start_detection()
            elif mode == 'background':
                self._reset_processors()
                self.processors = [BackgroundProcessor(self.cam)]
            elif mode == 'training':
                self._reset_processors()
                self.processors = [TrainingProcessor(self.cam, self.background, self.img_db, self.descriptor)]
            else:
                # invalid
                mode = self.settings['mode']

            self.settings['mode'] = mode

        if 'pause' in settings:
            self.settings['pause'] = settings['pause']
            if self.settings['pause']:
                self.cam.pause()
            else:
                self.cam.play()
        if 'action' in settings:
            action = settings['action']
            if action == 'complete_background' and self.settings['mode'] == 'background':
                self.processors[0].pause()
            if action == 'start_background' and self.settings['mode'] == 'background':
                self._reset_processors()
                self.processors = [BackgroundProcessor(self.cam)]
            elif action == 'set_rect' and self.settings['mode'] == 'training':
                self.processors[0].rect_index = int(self.settings['training_rect_index'])
            elif action == 'set_poly' and self.settings['mode'] == 'training':
                self.processors[0].poly_index = int(self.settings['training_poly_index'])
            elif action == 'label' and self.settings['mode'] == 'training':
                self.processors[0].capture(self.cam.get_frame(), self.settings['training_label'])
                # update our DB
                self.templates = [x.label for x in self.img_db]

        if 'descriptor' in settings:
            desc = settings['descriptor']
            if desc == 'BRISK':
                self.descriptor = cv2.BRISK_create()
            elif desc == 'AKAZE':
                self.descriptor = cv2.AKAZE_create()
            else:
                desc = self.settings['descriptor']
            self.settings['descriptor'] = desc

        # add our stream names now that everything has been added to the camera
        self.stream_names = self.cam.stream_names
        print(settings)

    async def update_state(self):
        if await self.cam.update():
            self.stream_number = len(self.cam.frame_processors) + 1
            #TODO: Insert update code here
            self.vision_state.time += 1
            return self.vision_state
        #cede control so other upates can happen
        await asyncio.sleep(0)
        return None

    async def update_loop(self):
        startTime = time.time()
        state = await self.update_state()
        if state is not None:
            await self.pub_sock.send_multipart(['vision-update'.encode(), state.SerializeToString()])
            #exponential moving average of update frequency
            self.frequency = self.frequency * 0.8 +  0.2 / (time.time() - startTime)


def init(video_filename, server_port, zmq_sub_port, zmq_pub_port, cc_hostname, template_dir, crop):
    c = Controller(zmq_sub_port, zmq_pub_port, cc_hostname)
    asyncio.ensure_future(c.handle_start(video_filename, server_port, template_dir, crop))
    loop = asyncio.get_event_loop()
    loop.run_forever()


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--video-filename', help='location of video or empty for webcam', default='', dest='video_filename')
    parser.add_argument('--server-port', help='port to run server', default='8888', dest='server_port')
    parser.add_argument('--zmq-sub-port', help='port for receiving zmq sub update', default=5000, dest='zmq_sub_port')
    parser.add_argument('--cc-hostname', help='hostname for cc to receive zmq pub updates', default='localhost', dest='cc_hostname')
    parser.add_argument('--zmq-pub-port', help='port for publishing my zmq updates', default=2400, dest='zmq_pub_port')
    parser.add_argument('--template-include', help='directory containing template images', dest='template_dir', required=True)
    parser.add_argument('--crop', help='two x,y points defining crop', dest='crop', nargs=4)

    args = parser.parse_args()
    if args.crop is not None:
        crop = [int(c) for c in args.crop]
    else:
        crop = None
    init(args.video_filename,
         args.server_port,
         args.zmq_sub_port,
         args.zmq_pub_port,
         args.cc_hostname,
         args.template_dir,
         crop)