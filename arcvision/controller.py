import zmq, time, argparse, asyncio, glob, os, sys, copy, json
import zmq.asyncio
from .server import start_server
from .utils import *
from multiprocessing import freeze_support
from .protobufs.graph_pb2 import Graph

from .projector import Projector
from .camera import Camera
from .background_processor import BackgroundProcessor
from .spatial_calibration_processor import SpatialCalibrationProcessor
from .dark_flow_detection_processor import DarkflowDetectionProcessor
from .detection_processor import DetectionProcessor
from .dark_flow_segment_processor import DarkflowSegmentProcessor
from .training_processor import TrainingProcessor

zmq.asyncio.install()

class Controller:
    '''Controls flow of reactor program'''
    def __init__(self, zmq_sub_port, zmq_pub_port, zmq_projector_port, cc_hostname):
        self.ctx = zmq.asyncio.Context()

        #subscribe to publishing socket
        zmq_uri = 'tcp://{}:{}'.format(cc_hostname, zmq_projector_port)
        print('Connecting pair Socket to {}'.format(zmq_uri))
        self.projector_sock = self.ctx.socket(zmq.PAIR)
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
        self.settings = {'mode': 'background',
                         'pause': False,
                         'descriptor': 'AKAZE',
                         'descriptor_threshold': 0.0002,
                         'descriptor_threshold_bounds': (0.00005,0.01),
                         'descriptor_threshold_step': 0.0005}
        self.modes = ['background',
                      'detection',
                      'darkflow',
                      'training',
                      'calibration']
        self.descriptors = ['AKAZE', 'SURF', 'BRISK' , 'KAZE']
        self.descriptor = cv2.AKAZE_create()#self.descriptor = cv2.xfeatures2d.SURF_create(400)#
        self.processors = []
        self.reserved_processors = []
        self.background = None

    def get_state_json(self):
        if self.settings['mode'] == 'training':
            try:
                self.settings['training_poly_len'] = self.processors[0].poly_len
                self.settings['training_rect_len'] = self.processors[0].rect_len
                self.settings['training_rect_index'] = self.processors[0].rect_index
                self.settings['training_poly_index'] = self.processors[0].poly_index
            except AttributeError:
                #somehow we can end up here before we finished reset processors
                #hack TODO: find out why
                pass
        return json.dumps(self.__dict__, default=lambda x: '')


    async def handle_start(self, video_filename, server_port, template_dir, output_video):
        '''Begin processing webcam and updating state'''
        self.cam = Camera(video_filename, output=output_video)
        self.img_db = ImageDB(template_dir)
        start_server(self.cam, self, server_port)
        print('Started arcvision server')
        self.projector_processor = None#Projector(self.cam, self.projector_sock)
        self.processors = []

        self.background = None
        self.background_processor = BackgroundProcessor(self.cam)
        self.transform_processor = SpatialCalibrationProcessor(self.cam, delay=8, stay=16, segmenter=DarkflowSegmentProcessor(self.cam))
        #self.transform_processor = SpatialCalibrationProcessor(self.cam, background=self.background)
        self.reserved_processors = [self.transform_processor]

        await self.update_settings(self.settings)

        while True:
            sys.stdout.flush()
            await self.update_loop()

    def _reset_processors(self):
        [x.close() for x in self.processors]
        self.processors = []
        self.background_processor.pause()
        self.transform_processor.pause()
        #self.projector_processor.transform = self.transform_processor.inv_transform

    def _start_detection(self):
        self.processors = [DetectionProcessor(self.cam, self.background,
                                              self.img_db, self.descriptor)]
    def _start_darkflow(self):
        self.processors = [DarkflowDetectionProcessor(self.cam, self.background)]

    async def update_settings(self, settings):

        status = 'settings_updated'
        if 'mode' in settings and settings['mode'] != self.settings['mode']:
            mode = settings['mode']
            if mode in self.modes:
                self.settings['mode'] = mode
            if mode == 'detection':
                self._reset_processors()
                self._start_detection()
            elif mode == 'darkflow':
                self._reset_processors()
                self._start_darkflow()
            elif mode == 'background':
                self._reset_processors()
                self.background_processor.reset()
            elif mode == 'calibration':
                self._reset_processors()
                self.transform_processor.reset()
                self.transform_processor.play()
            elif mode == 'training':
                self._reset_processors()
                self.processors = [TrainingProcessor(self.cam, self.img_db, self.descriptor, self.background)]
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
                self.background_processor.pause()
                self.transform_processor.background = self.background_processor.background
                self.background = self.background_processor.background
                self._reset_processors()
            if action == 'start_background' and self.settings['mode'] == 'background':
                self.background_processor.reset()
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
        #Set up the descriptor drop-downs
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
                    self.settings['descriptor_threshold'] = 0.00001
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
            elif desc == 'SURF':
                if settings['descriptor_threshold'] == 0:
                    self.settings['descriptor_threshold'] = 200
                else:
                    self.settings['descriptor_threshold'] = int(settings['descriptor_threshold'])
                self.descriptor = cv2.cv2.xfeatures2d.SURF_create(threshold=self.settings['descriptor_threshold'])
                self.settings['descriptor_threshold_bounds'] = (200, 10000)
                self.settings['descriptor_threshold_step'] = 50
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
        #print('vision state is ', self.vision_state)
        if state is not None:
            await self.pub_sock.send_multipart(['vision-update'.encode(), state.SerializeToString()])
            #exponential moving average of update frequency
            self.frequency = self.frequency * 0.8 +  0.2 / (time.time() - startTime)
            #print(self.frequency)

    def sync_objects(self):
        remove = []
        # use a new graph so we don't have to mark edges for deletion, just copy the nodes in
        newGraph = Graph()
        newGraph.time = self.vision_state.time
        for key in self.vision_state.nodes:
            oldNode = self.vision_state.nodes[key]
            node = newGraph.nodes[key]
            node.label = oldNode.label
            node.id = oldNode.id
            node.position[:] = oldNode.position[:]
            node.delete = oldNode.delete
            node.weight[:] = oldNode.weight[:]
        self.vision_state = newGraph
        for key in self.vision_state.nodes:
            # remove those that were marked last time
            if self.vision_state.nodes[key].delete:
                remove.append(key)
            elif self.vision_state.nodes[key].label == 'conditions':
                continue
            # mark others as dirty
            else:
                self.vision_state.nodes[key].delete = True

        # remove
        for r in remove:
            del self.vision_state.nodes[r]


        edgeIndex = 0
        # now update
        if not self.transform_processor.calibrate:
            processorsToUpdate = self.processors
        else:
            processorsToUpdate = self.processors + self.reserved_processors
        for p in processorsToUpdate:
            for o in p.objects:
                node = self.vision_state.nodes[o['id']]
                if o['label'] == 'conditions':
                    # don't bother setting position if this is the conditions node
                    node.position[:] = [0,0]
                    node.label = o['label']
                    node.id = o['id']
                    if 'weight' in o:
                        # weight is a list. append each value
                        if (len(o['weight']) == 0):
                            print("ERROR: conditions node has no weight")
                        for j in range(0,len(o['weight'])):
                            if len(node.weight) < j+1:
                                node.weight.append(o['weight'][j])
                            else:
                                node.weight[j] = o['weight'][j]
                    else:
                        print("ERROR: conditions node has no weight")
                    continue

                # don't transform while calibrating, otherwise it interferes
                elif not self.transform_processor.calibrate:
                    center_pos = copy.copy(o['center_scaled'])
                    node.position[:] = self.transform_processor.warp_point(center_pos)
                else:
                    node.position[:] = o['center_scaled']
                node.label = o['label']
                node.id = o['id']
                if 'weight' in o:
                    # weight is a list. append each value
                    for j in range(0,len(o['weight'])):
                        if len(node.weight) < j+1:
                            node.weight.append(o['weight'][j])
                        else:
                            node.weight[j] = o['weight'][j]
                node.delete = False

            # iterate through again, adding edges
            for o in p.objects:
                # check the number of connections this object is a primary for
                if ('connectedToPrimary' in o):
                    # there potentially are connections
                    for i in range(0,len(o['connectedToPrimary'])):
                        edge = self.vision_state.edges[edgeIndex]
                        edge.idA = o['id']
                        edge.labelA = o['label']
                        # the connections in connectedToPrimary is a list of tuples of the destination node, where the first is ID and second is label
                        dstId,dstLabel = o['connectedToPrimary'][i]
                        edge.idB = dstId
                        edge.labelB = dstLabel
                        edgeIndex += 1
                if ('connectedToSource' in o):
                    # the item is potentially connected to the source
                    if (o['connectedToSource'] == True):
                        #print("Alright, adding the edge to source")
                        edge = self.vision_state.edges[edgeIndex]
                        edge.idA = 0
                        edge.labelA = 'source'
                        edge.idB = o['id']
                        edge.labelB = o['label']
                        edgeIndex += 1




def init(video_filename, server_port, zmq_sub_port, zmq_pub_port, zmq_projector_port, cc_hostname, template_dir, output_video):
    c = Controller(zmq_sub_port, zmq_pub_port, zmq_projector_port, cc_hostname)
    asyncio.ensure_future(c.handle_start(video_filename, server_port, template_dir, output_video))
    loop = asyncio.get_event_loop()
    loop.run_forever()


def main():

    freeze_support()

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--video-filename', help='location of video or empty for webcam', default='', dest='video_filename')
    parser.add_argument('--server-port', help='port to run server', default='8888', dest='server_port')
    parser.add_argument('--zmq-sub-port', help='port for receiving zmq sub update', default=5000, dest='zmq_sub_port')
    parser.add_argument('--zmq-projector-port', help='port for connecting to projector', default=5001, dest='zmq_projector_port')
    parser.add_argument('--cc-hostname', help='hostname for cc to receive zmq pub updates', default='localhost', dest='cc_hostname')
    parser.add_argument('--zmq-pub-port', help='port for publishing my zmq updates', default=2400, dest='zmq_pub_port')
    parser.add_argument('--template-include', help='directory containing template images', dest='template_dir', required=True)
    parser.add_argument('--output-video', help='where to output video if desired', dest='output_video', default=None)
    parser.add_argument('--debug', help='enable async debugging tools', action='store_true')

    args = parser.parse_args()
    if args.debug:
        asyncio.get_event_loop().set_debug(True)
        import logging
        logging.getLogger('asyncio').setLevel(logging.DEBUG)

    init(args.video_filename,
         args.server_port,
         args.zmq_sub_port,
         args.zmq_pub_port,
         args.zmq_projector_port,
         args.cc_hostname,
         args.template_dir,
         args.output_video)