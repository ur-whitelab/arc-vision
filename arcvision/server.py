#!/usr/bin/env python3
'''Minimal sevrer to send/receive updates via HTTP instead of ZMQ'''


import tornado.web
import cv2
from tornado.platform.asyncio import AsyncIOMainLoop
import asyncio
import os
import json

AsyncIOMainLoop().install()

RESOURCES = os.path.join(os.path.dirname(__file__), os.pardir, 'resources')
WEB_STRIDE = 1

class HtmlPageHandler(tornado.web.RequestHandler):
    async def get(self, file_name='index.html'):
        # Check if page exists
        www = os.path.join(RESOURCES, file_name)
        if os.path.exists(www):
            # Render it
            self.render(www)
        else:
            # Page not found, generate template
            err_tmpl = tornado.template.Template("<html> Err 404, Page {{ name }} not found</html>")
            err_html = err_tmpl.generate(name=file_name)
            # Send response
            self.finish(err_html)

class StreamHandler(tornado.web.RequestHandler):
    def initialize(self, camera):
        self.camera = camera

    async def get(self, stream_name):
        '''
        Build MJPEG stream using the multipart HTTP header protocol
        '''
        # Set http header fields
        self.set_header('Cache-Control',
                        'no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0')
        self.set_header('Connection', 'close')
        self.set_header('Content-Type', 'multipart/x-mixed-replace;boundary=--boundarydonotcross')
        self.set_header( 'Pragma', 'no-cache')
        print('Received request, sending stream')

        while True:
            if self.request.connection.stream.closed():
                print('Request closed')
                return
            frame = self.camera.get_decorated_frame(stream_name)
            if frame is not None:
                ret, jpeg = cv2.imencode('.jpg', frame)
            else:
                ret = False
            img = ''
            if ret:
                img = jpeg.tostring()
                self.write("--boundarydonotcross\n")
                self.write("Content-type: image/jpeg\r\n")
                self.write("Content-length: %s\r\n\r\n" % len(img))
                self.write(img)
                await tornado.gen.Task(self.flush)

class StatsHandler(tornado.web.RequestHandler):
    '''Provides stats on speed of processing'''
    def initialize(self, controller):
        self.controller = controller

    async def get(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.write(json.dumps(self.controller.__dict__, default=lambda x: ''))


def start_server(camera, controller, port=8888):
    app = tornado.web.Application([
        (r"/",HtmlPageHandler),
        (r"/([a-z\-]+)/stream.mjpg", StreamHandler, {'camera': camera}),
        (r"/stats", StatsHandler, {'controller': controller})
    ])
    print('Starting server on port {}'.format(port))
    app.listen(port)


