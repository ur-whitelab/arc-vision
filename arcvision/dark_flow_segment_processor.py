from .utils import *

# from processor import Processor
from .processor import Processor


class DarkflowSegmentProcessor(Processor):
    def __init__(self, camera, stride=1, threshold=0.1):
        self.tfnet = load_darkflow('dot-tracking', gpu=1.0, threshold=threshold)
        super().__init__(camera, ['segment'], stride)

    def segments(self, frame):
        return self._process_frame(frame)

    def _process_frame(self, frame):
        result = self.tfnet.return_predict(frame) #get a dict of detected items with labels and confidences.
        sorted_result = sorted(result, key=lambda x: x['confidence'], reverse=True)
        segments = [darkflow_to_rect(x) for x in sorted_result]
        return segments

    async def process_frame(self, frame, frame_id):
        return

    async def decorate_frame(self, frame, name):
        if name == 'segment':
            for s in self.segments(frame):
                draw_rectangle(frame, s, (255, 255, 0), 1)
