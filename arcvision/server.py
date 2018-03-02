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
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
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
                #print('Frame was not None!')
                ret, jpeg = cv2.imencode('.jpg', frame)
            else:
                print('Frame WAS None! Oh NO')
                ret = False
            img = ''
            if ret:
                img = jpeg.tostring()
                self.write("--boundarydonotcross\n")
                self.write("Content-type: image/jpeg\r\n")
                self.write("Content-length: %s\r\n\r\n" % len(img))
                self.write(img)
                await tornado.gen.Task(self.flush)
            await asyncio.sleep(0.5)

class TemplateHandler(tornado.web.RequestHandler):
    '''Serves template images'''
    def initialize(self, controller):
        self.controller = controller

    async def get(self, template_name):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')

        templ = self.controller.img_db.get_img(template_name)

        if templ is not None:
            img = templ.img.copy()
            cv2.polylines(img, [templ.poly], True, (0,0,255), 2)
            ret, jpeg = cv2.imencode('.jpg', img)
            if ret:
                self.write("Content-type: image/jpeg\r\n")
                self.write("Content-length: %s\r\n\r\n" % len(img))
                self.write(jpeg.tostring())

class StatsHandler(tornado.web.RequestHandler):
    '''Provides info about controller processing'''
    def initialize(self, controller):
        self.controller = controller

    async def get(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.write(self.controller.get_state_json())

class SettingsHandler(tornado.web.RequestHandler):
    def initialize(self, controller):
        self.controller = controller

    async def post(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        if len(self.request.body) > 1:
            new_settings = json.loads(self.request.body.decode())
            response = await self.controller.update_settings(new_settings)
            self.write(response)


def start_server(camera, controller, port=8888):

    app = tornado.web.Application([
        (r"/",HtmlPageHandler),
        (r"/stream/([A-Za-z\-]+).mjpg", StreamHandler, {'camera': camera}),
        (r"/stats", StatsHandler, {'controller': controller}),
        (r"/settings", SettingsHandler, {'controller': controller}),
        (r"/template/(a-z\-])+", TemplateHandler, {'controller': controller})
    ])
    print('Starting server on port {}'.format(port))
    app.listen(port)


