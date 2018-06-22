import cv2
from .utils import *

from .processor import Processor
from .tracker_processor import TrackerProcessor


class DarkflowDetectionProcessor(Processor):
    '''Detects query images in frame. Uses async to spread out computation. Cannot handle replicas of an object in frame'''
    def __init__(self, camera, background, stride=3,
                 threshold=0.1, track=True):
        self.tfnet = load_darkflow('reactor-tracking', gpu=1.0, threshold=threshold)
        self.id_i = 1000#skip over 0 thru 999
        #we have a specific order required
        #set-up our tracker
        # give estimate of our stride
        if track:
            self.tracker = TrackerProcessor(camera, stride,  background,  delete_threshold_period = 2, do_tracking = False)
        else:
            self.tracker = None

        #then us
        super().__init__(camera, ['identify'], stride)
        self.track = track
        self.stride = stride

    @property
    def objects(self):
        if self.tracker is None:
            return []
        return self.tracker.objects

    def close(self):
        super().close()
        self.tracker.close()

    async def process_frame(self, frame, frame_ind):
        result = self.tfnet.return_predict(frame)#get a dict of detected items with labels and confidences.
        for item in result:
            brect = darkflow_to_rect(item)
            label = item['label']
            id_num = self.id_i
            new_obj = self.tracker.track(frame, brect, None, label, id_num)
            if(new_obj):
                self.id_i += 1

        return

    async def decorate_frame(self, frame, name):
        for i,item in enumerate(self.tracker._tracking):
            draw_rectangle(frame, item['brect'], (255, 0, 0), 1)
            (x,y) = rect_center(item['brect'])
            cv2.circle(frame, (int(x),int(y)), 10, (0,0,255), -1)
            cv2.putText(frame,
                        '{}: {}'.format(item['name'], item['observed']),
                        (0, 60 * (i+1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255))
        return frame
