import cv2, glob, pickle, os, copy, hashlib, math, pkg_resources
import numpy as np
from darkflow.net.build import TFNet

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
            self.meta_id = 0

        @property
        def id(self):
            return self.meta_id
        def __getstate__(self):
            # remove keypoints from pickle
            odict = copy.copy(self.__dict__)
            del odict['keypoints']
            del odict['features']
            return odict

        def set_id(self, number):
            self.meta_id = number

        def __setstate__(self, odict):
            self.__dict__.update(odict)
            self.keypoints = None
            self.features = None

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
            idNumber = 1
            for i in glob.glob('**/*.pickle', recursive=True):
                with open(os.path.join(template_dir, i), 'rb') as f:
                    img = pickle.load(f)
                    img.set_id(idNumber)
                    print("Image {} has ID {}".format(img.label, img.id))
                    idNumber += 1
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
            rel_path = hashlib.sha256(np.array_repr(img).encode()).hexdigest()
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
    return tuple(rect)

def rect_view(frame, rect):
    '''Use a bounding rectangle to create a view of a frame'''
    if rect[3] * rect[2] < 20:
        raise ValueError('Attempting to create too small of a view')
    return frame[ rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2]) ]

def keypoints_view(desc, frame, rect):
    '''return the keypoints limited to a region'''
    rect = stretch_rectangle(rect, frame)#embiggen the rectangle by 1.2x

    frame_view = rect_view(frame, rect)#restrict area of interest to the embiggened rectangle
    kp, des = desc.detectAndCompute(frame_view,None)#get the keypoints in that region
    #need to transform the key points back
    for i in range(len(kp)):
        kp[i].pt = (rect[0] + kp[i].pt[0], rect[1] + kp[i].pt[1])
    return kp, des

def draw_rectangle(frame, rect, *args):
    rect = [int(r) for r in rect]
    cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), *args)


def intersecting_rects(a, b):
    dx = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    dy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if (dx >= 0) and (dy >= 0):
        return True
    return False

def scale_point(point, frame):
    '''Takes in a point as a tuple of ints and returns a list of floats in scaled coordinates (0 to 1)'''
    x = float(point[0])/frame.shape[1]
    y = float(point[1])/frame.shape[0]
    return [x,y]

def box_scaled_center(rect, frame):
    #EXPECTS (topleftx, toplefty, bottomrightx, bottomrighty)
    x = (rect[0] + (rect[2] - rect[0]) / 2) / frame.shape[1]
    y = (rect[1] + (rect[3] - rect[1]) / 2) / frame.shape[0]
    return [x,y]

def rect_scaled_center(rect, frame):
    #EXPECTS (topleftx, toplefty, bottomrightx, bottomrighty)
    x = (rect[0] + rect[2] / 2) / frame.shape[1]
    y = (rect[1] + rect[3] / 2) / frame.shape[0]
    return [x,y]

def rect_center(rect):
      #EXPECTS (topleftx, toplefty, bottomrightx, bottomrighty)
    x = (rect[0] + rect[2] / 2)
    y = (rect[1] + rect[3] / 2)
    return [x,y]

def poly_scaled_center(polygon, frame):
    #Find the scaled center of a cv2 polygon.
    #taken from https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/
    moment = cv2.moments(polygon)
    centerX = moment['m10'] / moment['m00'] / frame.shape[1]
    centerY = moment['m01'] / moment['m00'] / frame.shape[0]
    return [centerX, centerY]

def box_area(rect):
    return (abs(rect[0]- rect[2])* abs(rect[1] -rect[3]))

def rect_area(rect):
    return abs(rect[2] * rect[3])


def rect_color_channel(frame, rect):
    '''Returns which channel is maximum'''
    return np.argmax(np.mean(rect_view(frame, rect), axis=(0,1)))

def percent_diff(existingItem, newItem):
    #TODO: fix so this is robust to 'existingItem' being near-zero, or move to some other comparison
    return float(newItem-existingItem)/existingItem

def rect_to_endpoints(rect):
    '''Lazy implementation of finding endpoints of a bounding rectangle.  Find the vertices, use the lowest and its hypoteneuse'''
    box = np.int0(cv2.boxPoints(rect))
    return (box[0],box[2])


def line_from_endpoints(endpoints):
    #TODO: idea: work with angles instead, take actual diff, not relative...
    ''' Compute the slope and bias of a line function given 2 endpoints'''
    endpoint1 = endpoints[0]
    endpoint2 = endpoints[1]
    if ((endpoint1[0] - endpoint2[0] == 0)):
        slope = np.inf
    elif(endpoint1[1] - endpoint2[1] == 0):
        slope = 0.0
    else:
        if(endpoint1[0] > endpoint2[0]):#reverse because of back-projection...
            sign = 1.0
        else:
            sign = -1.0
        #due to how line endpoints are given (lowest y-val is always first), we have to be careful with slopes.
        slope = sign * abs(endpoint1[1] - endpoint2[1])/abs(endpoint1[0] - endpoint2[0])
    intercept = endpoint1[1] - slope*endpoint1[0]
    return (slope,intercept)

''' Calculate Euclidean distance between two endpoints GIVEN AS A TUPLE'''
def distance_pts(endpoints):
    endpoint1 = endpoints[0]
    endpoint2 = endpoints[1]
    return math.sqrt(math.pow(endpoint1[0]-endpoint2[0],2) + math.pow(endpoint1[1]-endpoint2[1],2))

def val_in_range(val, lower_bound,upper_bound):
    return ((val >= lower_bound) and (val <= upper_bound))

def load_darkflow(directory, threshold=0.2, **args):
    resource_path =pkg_resources.resource_filename(__name__, 'resources/models/' + directory)
    options = args
    options['threshold'] = threshold
    try:
        options['pbLoad'] = list(glob.glob(resource_path + '/*.pb'))[0]
        options['metaLoad'] = list(glob.glob(resource_path + '/*.meta'))[0]
    except IndexError:
        raise FileNotFoundError(f'Could not find darkflow model pb or meta in {resource_path}')
    return TFNet(options)

def darkflow_to_box(df):
    #opencv-style bboxes are [x1, y1, x2, y2] but it's top-down for y, and frame.shape[0] is y-shape
    bbox = [ df['topleft']['x'], df['topleft']['y'], df['bottomright']['x'],df['bottomright']['y'] ]
    return bbox

def darkflow_to_rect(df):
    rect = [ df['topleft']['x'], df['topleft']['y'], df['bottomright']['x'] - df['topleft']['x'], df['bottomright']['y'] - df['topleft']['y'] ]
    return rect

def diff_blur(frame1, frame2, do_sum=True):
    img = cv2.absdiff(frame1, frame2)
    if do_sum:
        img = np.sum(img, 2).astype(np.uint8)
    img = cv2.medianBlur(img, 9)
    return img