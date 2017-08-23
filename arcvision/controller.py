import zmq
import zmq.asyncio
import time
import argparse
import asyncio
import glob
import os
from .camera import Camera
from .server import start_server
from .calibration import Calibrate
from .processor import *
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
        self.stream_number = 1

        #create state
        self.vision_state = ReactorSystem()
        self.vision_state.time = 0


    async def handle_start(self, video_filename, server_port, template_dir):
        '''Begin processing webcam and updating state'''

        self.cam = Camera(video_filename)
        start_server(self.cam, self, server_port)
        print('Started arcvision server')
        import sys
        sys.stdout.flush()

        PreprocessProcessor(self.cam)
        ExampleProcessor(self.cam)
        #load images
        paths = []
        labels = []
        original = os.getcwd()
        template_dir = os.path.abspath(template_dir)
        try:
            os.chdir(template_dir)
            print('Found these template images in {}:'.format(template_dir))
            for i in glob.glob('**/*.jpg', recursive=True):
                labels.append(i)
                paths.append(os.path.join(template_dir, i))
                print('\t' + labels[-1])

        finally:
            os.chdir(original)

        DetectionProcessor(self.cam, labels=labels, query_images=paths)
        BackgroundProcessor(self.cam)

        while True:
            await self.update_loop()

    async def update_state(self):
        if await self.cam.update():
            self.stream_number = len(self.cam.frame_processors) + 1
            #TODO: Insert update code here
            self.vision_state.time += 1
            return self.vision_state
        return None

    async def update_loop(self):
        startTime = time.time()
        state = await self.update_state()
        if state is not None:
            await self.pub_sock.send_multipart(['vision-update'.encode(), state.SerializeToString()])
            #exponential moving average of update frequency
            self.frequency = self.frequency * 0.8 +  0.2 / (time.time() - startTime)

def init(video_filename, server_port, zmq_sub_port, zmq_pub_port, cc_hostname, template_dir):
    c = Controller(zmq_sub_port, zmq_pub_port, cc_hostname)
    asyncio.ensure_future(c.handle_start(video_filename, server_port, template_dir))
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
    args = parser.parse_args()
    init(args.video_filename, args.server_port, args.zmq_sub_port, args.zmq_pub_port, args.cc_hostname, args.template_dir)