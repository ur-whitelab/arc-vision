import cv2
import glob
import pickle
import os

class ImageDB:
    '''Class which stores pre-processed, labeled images used in identification'''

    class Image:
        '''POD store of image data'''
        def __init__(self, path, img, label, poly, keypoints = None):
            self.img = img
            self.label = label
            self.poly = poly
            self.path = path
            self.keypoints = keypoints
            self.features = None
            self.color = (255,255,255)
        def __getstate__(self):
            # remove keypoints from pickle
            odict = self.__dict__
            del odict['keypoints']
            del odict['features']
            return odict

    # --- end Image Class

    def __init__(self, template_dir, load=True):
        self.images = []
        self.template_dir = template_dir
        # load any images
        if load:
            self._load(template_dir)


    def _load(self, template_dir):
        original = os.getcwd()
        template_dir = os.path.abspath(template_dir)
        try:
            os.chdir(template_dir)
            print('Found these pre-processed images in {}:'.format(template_dir))
            for i in glob.glob('**/*.pickle', recursive=True):
                with open(os.path.join(template_dir, i), 'rb') as f:
                    img = pickle.load(f)
                    self.images.append(img)
                    print('\t' + i)

        finally:
            os.chdir(original)

    def __iter__(self):
        return self.images.__iter__()

    def __len__(self):
        return len(self.images)

    def get_img(self, label):
        return filter(lambda s: s.label == label, self.images)

    def set_descriptor(self, descriptor):
        for img in self:
            img.keypoints = descriptor.detect(img.img)


    def store_img(self, img, label, poly, keypoints = None, processed_img = None, rel_path = None):
        '''
            img: the image
            poly: polygon points
            path: path ending with name (not extension) which will be used for prepending pickle, processed, etc
            label: name
            processed_img: the processed image. Will be saved for reference
        '''
        if rel_path is None:
            rel_path = label
        path = os.path.join(self.template_dir, rel_path)
        if len(path.split('.jpg')) > 1:
            path = path.split('.jpg')[0]
        img = ImageDB.Image(path, img, label, poly, keypoints)
        self.images.append(img)

        if processed_img is not None:
            cv2.imwrite(path + '_processes.jpg', processed_img)

        # store pickle
        with open(path + '.pickle', 'wb') as f:
            pickle.dump(img, f)


def stretch_rectangle(rect, frame, stretch=1.2):
    # stretch out the rectangle
    rect = list(rect)
    rect[0] += int(rect[2] * (1 - stretch) // 2)
    rect[1] += int(rect[3] * (1 - stretch) // 2)
    rect[2] = int(rect[2] * stretch)
    rect[3] = int(rect[3] * stretch)

    rect[0] = max(rect[0], 0)
    rect[1] = max(rect[1], 0)
    rect[2] = min(frame.shape[1], rect[2])
    rect[3] = min(frame.shape[0], rect[3])
    return rect

def rect_view(frame, rect):
    '''Use a bounding rectangle to create a view of a frame'''
    if rect[3] <= 3 or rect[2] <= 3:
        raise ValueError('Attempting to create too small of a view')
    return frame[ rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2]) ]

def keypoints_view(desc, frame, rect):
    '''return the keypoints limited to a region'''
    rect = stretch_rectangle(rect, frame)

    frame_view = rect_view(frame, rect)
    kp, des = desc.detectAndCompute(frame_view,None)
    #need to transform the key points back
    for i in range(len(kp)):
        kp[i].pt = (rect[0] + kp[i].pt[0], rect[1] + kp[i].pt[1])
    return kp, des

def draw_rectangle(frame, rect, *args):
    cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), *args)

