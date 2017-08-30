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

from .protobufs.graph_pb2 import Graph


zmq.asyncio.install()

class Controller:
    '''Controls flow of reactor program'''
    def __init__(self, zmq_sub_port, zmq_pub_port, zmq_projector_port, cc_hostname):
        self.ctx = zmq.asyncio.Context()

        #subscribe to publishing socket
        zmq_uri = 'tcp://{}:{}'.format(cc_hostname, zmq_projector_port)
        print('Connecting REQ Socket to {}'.format(zmq_uri))
        self.projector_sock = self.ctx.socket(zmq.REQ)
        self.projector_sock.connect(zmq_uri)

        #register publishing socket
        zmq_uri = 'tcp://{}:{}'.format(cc_hostname, zmq_pub_port)
        print('Connecting PUB Socket to {}'.format(zmq_uri))
        self.pub_sock = self.ctx.socket(zmq.PUB)
        self.pub_sock.connect(zmq_uri)

        #statistics
        self.frequency = 1
        self.stream_names = []

        #create state
        self.vision_state = Graph()
        self.vision_state.time = 0

        #settings
        self.settings = {'mode': 'background', 'pause': False, 'descriptor': 'BRISK', 'descriptor_threshold': 30, 'descriptor_threshold_bounds': (1,30), 'descriptor_threshold_step': 1}
        self.modes = ['background', 'detection', 'training', 'extent']
        self.descriptors = ['BRISK', 'AKAZE', 'KAZE']
        self.descriptor = cv2.BRISK_create()
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

        self.cam = Camera(self.projector_sock, video_filename)
        self.img_db = ImageDB(template_dir)
        start_server(self.cam, self, server_port)
        print('Started arcvision server')

        self.crop_processor = CropProcessor(self.cam, crop)
        #PreprocessProcessor(self.cam)
        self.processors = [BackgroundProcessor(self.cam)]

        await self.update_settings(self.settings)

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
        self.processors = [DetectionProcessor(self.cam, self.background,
                                              self.img_db, self.descriptor)]

    async def update_settings(self, settings):

        status = 'settings_updated'
        if 'mode' in settings and settings['mode'] != self.settings['mode']:
            mode = settings['mode']
            if mode in self.modes:
                self.settings['mode'] = mode
            if mode == 'detection':
                self._reset_processors()
                self._start_detection()
            elif mode == 'background':
                self._reset_processors()
                self.processors = [BackgroundProcessor(self.cam)]
            elif mode == 'training':
                self._reset_processors()
                self.processors = [TrainingProcessor(self.cam, self.background, self.img_db, self.descriptor)]
            elif mode == 'extent':
                self._reset_processors()
                self.processors = [SegmentProcessor(self.cam, self.background, 16, 1, 1)]
            # notify that our mode changed
            await self.pub_sock.send_multipart(['vision-mode'.encode(), mode.encode()])

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
                self.processors[0].rect_index = int(settings['training_rect_index'])
            elif action == 'set_poly' and self.settings['mode'] == 'training':
                self.processors[0].poly_index = int(settings['training_poly_index'])
            elif action == 'label' and self.settings['mode'] == 'training':
                if self.processors[0].capture(self.cam.get_frame(), settings['training_label']):
                    status = 'label_set'
                else:
                    status = 'label_fail'

                # update our DB
                self.templates = [x.label for x in self.img_db]
            elif action == 'set_extent' and self.settings['mode'] == 'extent':
                extent_view = None
                try:
                    extent_view = next(self.processors[0].segments())
                except StopIteration:
                    pass
                # set the extent
                self.crop_processor.rect = extent_view
                # need to fix the background dimensions
                self.background = rect_view(self.background, extent_view)
                self.processors[0].background = self.background
            elif action == 'reset_extent' and self.settings['mode'] == 'extent':
                # unset extent
                self.crop_processor.rect = None
                # we lost background, so switch modes
                self.background = np.zeros(self.cam.get_frame().shape, np.uint8)
                self.processors[0].background = self.background

        if 'descriptor' in settings and (settings['descriptor'] != self.settings['descriptor'] or settings['descriptor_threshold'] != self.settings['descriptor_threshold']):
            desc = settings['descriptor']
            if desc == 'BRISK':
                if settings['descriptor_threshold'] == 0:
                    self.settings['descriptor_threshold'] = 30
                else:
                    self.settings['descriptor_threshold'] = int(settings['descriptor_threshold'])
                self.descriptor = cv2.BRISK_create(thresh=self.settings['descriptor_threshold'])
                self.settings['descriptor_threshold_bounds'] = (1, 50)
                self.settings['descriptor_threshold_step'] = 1
            elif desc == 'AKAZE':
                if settings['descriptor_threshold'] == 0:
                    self.settings['descriptor_threshold'] = 0.001
                else:
                    self.settings['descriptor_threshold'] = float(settings['descriptor_threshold'])
                self.descriptor = cv2.AKAZE_create(threshold=self.settings['descriptor_threshold'])
                self.settings['descriptor_threshold_bounds'] = (0.00005, 0.005)
                self.settings['descriptor_threshold_step'] = 0.00005
            elif desc == 'KAZE':
                if settings['descriptor_threshold'] == 0:
                    self.settings['descriptor_threshold'] = 0.001
                else:
                    self.settings['descriptor_threshold'] = float(settings['descriptor_threshold'])
                self.descriptor = cv2.KAZE_create(threshold=self.settings['descriptor_threshold'])
                self.settings['descriptor_threshold_bounds'] = (0.00005, 0.005)
                self.settings['descriptor_threshold_step'] = 0.00005
            else:
                desc = self.settings['descriptor']

            self.settings['descriptor'] = desc
            if self.settings['mode'] == 'training':
                self.processors[0].set_descriptor(self.descriptor)
            elif self.settings['mode'] == 'detection':
                self.processors[0].set_descriptor(self.descriptor)
            self.img_db.set_descriptor(self.descriptor)

        # add our stream names now that everything has been added to the camera
        self.stream_names = self.cam.stream_names
        self.stream_number = len(self.cam.frame_processors) + 1
        print(settings)
        return status

    async def update_state(self):
        if await self.cam.update():

            self.vision_state.time += 1
            self.sync_objects()
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

    def sync_objects(self):
        remove = []
        for key in self.vision_state.nodes:
            # remove those that were marked last time
            if self.vision_state.nodes[key].delete:
                remove.append(key)
            # mark others as dirty
            else:
                self.vision_state.nodes[key].delete = True

        # remove
        for r in remove:
            del self.vision_state.nodes[r]

        # now update
        for o in self.processors[0].objects:
            node = self.vision_state.nodes[o['id']]
            node.position[:] = o['center_scaled']
            node.label = o['label']
            node.id = o['id']
            node.delete = False


def init(video_filename, server_port, zmq_sub_port, zmq_pub_port, zmq_projector_port, cc_hostname, template_dir, crop):
    c = Controller(zmq_sub_port, zmq_pub_port, zmq_projector_port, cc_hostname)
    asyncio.ensure_future(c.handle_start(video_filename, server_port, template_dir, crop))
    loop = asyncio.get_event_loop()
    loop.run_forever()


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--video-filename', help='location of video or empty for webcam', default='', dest='video_filename')
    parser.add_argument('--server-port', help='port to run server', default='8888', dest='server_port')
    parser.add_argument('--zmq-sub-port', help='port for receiving zmq sub update', default=5000, dest='zmq_sub_port')
    parser.add_argument('--zmq-projector-port', help='port for connecting to projector', default=5001, dest='zmq_projector_port')
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
         args.zmq_projector_port,
         args.cc_hostname,
         args.template_dir,
         crop)