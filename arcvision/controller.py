import zmq
import zmq.asyncio
import time
import argparse
import asyncio
from .camera import Camera
from .server import start_server
from .tracking import Detector
from .calibration import Calibrate


zmq.asyncio.install()

class Controller:
    '''Controls flow of vision program'''
    def __init__(self, zmq_uri):
        self.ctx = zmq.asyncio.Context()
        print('Opening PUB Socket on {}'.format(zmq_uri))
        self.psock = self.ctx.socket(zmq.PUB)
        self.psock.bind(zmq_uri)
        self.state = 'Placeholder'
        self.frequency = 1

    async def handle_start(self, video_filename, server_port):
        '''Begin processing webcam and updating state'''

        self.cam = Camera(video_filename)
        start_server(self.cam, self, server_port)
        print('Started arcvision server')
        import sys
        sys.stdout.flush()

        #run all of calibration here

        c = Calibrate()
        c.calibrate_image(self.cam.get_frame())

        d = Detector(self.cam)

        print('here')
        d.get_snapshot(self.cam,file_location='temp/background.png')
        print('done')
        d.attach(self.cam)
        while True:
            await self.update_loop()

    async def update_state(self):
        if await self.cam.update():
            #TODO: Insert update code here
            self.state = 'Placeholder'
            return self.state
        return None

    async def update_loop(self):
        startTime = time.time()
        state = await self.update_state()
        if state is not None:
            await self.psock.send_multipart(['update'.encode(), state.encode()])
            #exponential moving average of update frequency
            self.frequency = self.frequency * 0.8 +  0.2 / (time.time() - startTime)

def init(video_filename, server_port, zmq_port, hostname):
    c = Controller('tcp://{}:{}'.format(hostname, zmq_port))
    asyncio.ensure_future(c.handle_start(video_filename, server_port))
    loop = asyncio.get_event_loop()
    loop.run_forever()


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--video-filename', help='location of video or empty for webcam', default='', dest='video_filename')
    parser.add_argument('--server-port', help='port to run streaming server', default='8888', dest='server_port')
    parser.add_argument('--zmq-port', help='port for pub/sub zeromq', default=5000, dest='zmq_port')
    parser.add_argument('--hostname', help='hostname for pub/sub zeromq', default='*')
    args = parser.parse_args()
    init(args.video_filename, args.server_port, args.zmq_port, args.hostname)