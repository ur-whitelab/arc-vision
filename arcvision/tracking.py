import asyncio

class Detector:
    '''Class that detects multiple objects given a set of labels and images'''
    def __init__(self, query_images=[], labels=None):
        if labels is None:
            labels = ['Key {}'.format(i) for i in range(len(query_images))]
        self.keys = [{'name': k, 'img': cv2.imread(i, 0), 'path': i} for k,i in zip(labels, query_images)]

    def frame_update(self, frame):
        '''Perform update on frame, carrying out algorithm'''
        #simulate working hard
        asyncio.sleep(0.25)

    def frame_decorate(self, frame):
        '''Draw visuals onto the given frame, without carrying-out update'''
        pass

    def attach(self, camera, stride=1):
        camera.add_frame_fxn(self.frame_update, stride)


