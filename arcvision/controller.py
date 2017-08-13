import zmq
import zmq.asyncio
import time
import fire
import asyncio
from .camera import Camera
from .server import start_server
from .tracking import Detector


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
        print('Received start trigger. Opening camera')
        self.cam = Camera(video_filename)
        start_server(self.cam, self, server_port)
        d = Detector()
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

def main(video_filename=0, server_port=8888, zmq_port=5000, hostname='*'):
    c = Controller('tcp://{}:{}'.format(hostname, zmq_port))
    asyncio.ensure_future(c.handle_start(video_filename, server_port))
    loop = asyncio.get_event_loop()
    loop.run_forever()


if __name__ == '__main__':
    fire.Fire(main)