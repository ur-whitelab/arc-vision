#!/usr/bin/env python3

import fire
import asyncio
from .camera import Camera
from .server import start_server

def main(video_filename=0, server_port=8888):
    '''Open webcam and begin processing vision'''
    cam = Camera(video_filename)
    start_server(cam, server_port)
    loop = asyncio.get_event_loop()
    asyncio.ensure_future(cam.start())
    loop.run_forever()


if __name__ == '__main__':
    fire.Fire(main)