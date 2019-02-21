import numpy as np
from numpy import dot as dot
from numpy.linalg import inv as inverse

import sys, cv2
from .utils import *
import time

from .processor import Processor
from .line_detection_processor import LineDetectionProcessor
from .dial_processor import DialProcessor

class TrackerProcessor(Processor):

    @property
    def objects(self):
        '''Objects should have a dictionary with center, brect, name, and id'''
        if (self.dialReader is not None):
            return self.dialReader._objects + self._tracking
        else:
            return self._tracking


    def __init__(self, camera, detector_stride, background, delete_threshold_period=1.0, stride=2, detectLines = True, readDials = True, do_tracking = True, alpha=0.8):
        super().__init__(camera, ['track','line-segmentation'], stride)
        self._tracking = []
        self.do_tracking = do_tracking #this should only be False if we're using darkflow
        self.alpha = alpha #this is the spring constant
        self.labels = {}
        self.stride = stride
        self.ticks = 0
        if(do_tracking):
            self.optflow = cv2.DualTVL1OpticalFlow_create()#use dense optical flow to track
        self.detect_interval = 3
        self.prev_gray = None
        self.tracks = []
        self.min_pts_near = 4#the minimum number of points we need to say an object's center is here
        self.pts_dist_squared_th = int(75.0 / 2 / 720.0 * background.shape[0])**2
        self.feature_params = dict( maxCorners = 500,
                qualityLevel = 0.3,
                minDistance = 7,
                blockSize = 7 )
        print('initializing trackerprocessor. background.shape is {} by {}'.format(background.shape[0], background.shape[1]))
        self.dist_th_upper = int(150.0 / 720.0 * background.shape[0])# distance upper threshold, in pixels
        self.dist_th_lower = int(75.0 / 720.0 * background.shape[0]) # to account for the size of the reactor
        print('dist_th_upper is {} and dist_th_lower is {}'.format(self.dist_th_upper, self.dist_th_lower))
        self.max_obs_possible = 24
        # set up line detector
        if detectLines:
             self.lineDetector = LineDetectionProcessor(camera,stride,background)
        else:
            self.lineDetector = None
        # need to keep our own ticks because
        # we don't know frame index when track() is called
        if detector_stride > 0:
            self.ticks_per_obs = detector_stride * delete_threshold_period /self.stride

        if readDials:
            self.dialReader = DialProcessor(camera, stride=1)
        else:
            self.dialReader = None

    def close(self):
        super().close()
        if self.dialReader is not None:
            self.dialReader.close()

    async def process_frame(self, frame, frame_ind):
        self.ticks += 1
        delete = []

        if(self.do_tracking):
            smaller_frame = frame
            smaller_frame = smaller_frame#4x downsampling
            smaller_frame = cv2.cvtColor(smaller_frame, cv2.COLOR_BGR2GRAY)
            gray = smaller_frame#cv2.UMat(smaller_frame)
            if(self.prev_gray is None):
                self.prev_gray = gray#gray
                return
            img0, img1 = self.prev_gray, gray#gray
            #p0 = np.float32(self.tracks).reshape(-1, 1, 2)\
            p1 = self.optflow.calc(img0, img1, None)#cv2.calcOpticalFlowFarneback(img0, img1, None, 0.5, 2, 15, 2, 5, 1.1, 0)#, p0)#, None, **self.lk_params)  p1, _st, _err
            if(frame_ind % self.detect_interval == 0 or len(self.tracks)==0):
                mask = np.zeros((smaller_frame.shape), dtype=np.uint8)#np.zeros_like(gray)
                mask[:] = 255
                self.tracks = np.float32(cv2.goodFeaturesToTrack(smaller_frame, mask=mask, **self.feature_params)).reshape(-1,2)

        for i,t in enumerate(self._tracking):
            old_center = t['center_scaled']
            t['connectedToPrimary'] = [] # list of tracked objects it is connected to as the primary/source node
            t['connectedToSecondary'] = []
            t['connectedToSource'] = False
            #status,brect = t['tracker'].update(umat_frame)
            t['observed'] -= 1
            if(self.do_tracking):
                # we know our objects should stay the same size all of the time.
                # check if the size dramatically changed.  if so, the object most likely was removed
                # if not, rescale the tracked brect to the correct size
                #print("t['center_scaled'] is {}".format(t['center_scaled']))
                center_unscaled = (t['center_scaled'][0]*smaller_frame.shape[1] , t['center_scaled'][1]*smaller_frame.shape[0])
                #print('center_unscaled is {} and smaller_frame.shape is {}'.format(center_unscaled, smaller_frame.shape))
                #print('the dimensions of p1 are {}'.format(p1.shape))
                a = int(center_unscaled[1])
                b = int(center_unscaled[0])
                flow_at_center = [p1[a][b][0], p1[a][b][1]]#get the flow computed at previous center of object
                #flow_at_center = flow_at_center[::-1]#this is reversed for some reason..?
                flow_at_center = scale_point(flow_at_center, smaller_frame)
                print('flow_at_center is {}'.format(flow_at_center))
                dist = distance_pts([[0,0], flow_at_center ])#this is the magnitude of the vector
                # check if its new location is a reflection, or drastically far away
                near_pts = 0
                for pt in self.tracks:
                    if(distance_pts([center_unscaled, pt]) <= self.pts_dist_squared_th):
                        near_pts += 1
                if (dist < .05 * max(smaller_frame.shape) and near_pts >= 5):#don't move more than 5% of the biggest dimension
                    #print('Updated distance is {}'.format(dist))
                    # rescale the brect to match the original area?
                    t['center_scaled'][0] += flow_at_center[0]
                    t['center_scaled'][1] += flow_at_center[1]
                    t['observed'] = min(t['observed'] +2, self.max_obs_possible)
                #put note about it
            # check obs counts
            if t['observed'] < 0:
                delete.append(i)
        offset = 0
        delete.sort()
        for i in delete:
            for j,t in enumerate(self._tracking):
                # remove any references of this node from connectedToPrimary
                t2 = self._tracking[i-offset]
                if (t2['id'],t2['label']) in t['connectedToPrimary']:
                    index = t['connectedToPrimary'].index((t2['id'],t2['label']))
                    del t['connectedToPrimary'][index]
            del self._tracking[i - offset]
            offset += 1

        #update _tracking with the connections each object has
        await self._connect_objects(frame.shape)
        #f frame_ind % 4 * self.stride == 0:
        #    for t in self._tracking:
        #        print('{} is connected to ({})'.format(t['label'], t['connectedToPrimary']))
                #print('Is {} connected to the feed source? {}'.format(t['label'], t['connectedToSource']))
        if(self.do_tracking):
            self.prev_gray = gray
        return

    async def _connect_objects(self, frameSize):
        if (self.lineDetector is None) or len(self.lineDetector.lines) == 0:
            return
        ''' Iterates through tracked objects and the detected lines, finding objects are connected. Updates self._tracking to have directional knowledge of connections'''
        source_position_scaled = (1.0,0.5)#first coord is X from L to R, second coord is Y from TOP to BOTTOM
        source_position_unscaled = (frameSize[1],round(frameSize[0]*.5))
        #source_position_unscaled = self._unscale_point(source_position_scaled, frameSize)
        source_dist_thresh_upper = int(200.0 / 720.0 * frameSize[0])
        source_dist_thresh_lower = int(10.0 / 720.0 * frameSize[0])
        #print('source_dist_thresh_upper is {} and framesize[1] is {}'.format(source_dist_thresh_upper, frameSize[1]))
        used_lines = []
        for i,t1 in enumerate(self._tracking):
            center = self._unscale_point(t1['center_scaled'], frameSize)
            # find all lines that have an endpoint near the center of this object
            for k,line in enumerate(self.lineDetector.lines):
                if k in used_lines:
                    continue # dont attempt to use this line if it is already associated with something

                dist_ep1 = distance_pts((center, line['endpoints'][0]))
                dist_ep2 = distance_pts((center, line['endpoints'][1]))
                nearbyEndpointFound = False
                #print('Distances for {} {} (position {}) to the endpoints are {} (position {}) and {} (position {})'.format(t1['label'], t1['id'], center, min(dist_ep1,dist_ep2), line['endpoints'][0], max(dist_ep1,dist_ep2), line['endpoints'][1]))
                if (val_in_range(dist_ep1,self.dist_th_lower,self.dist_th_upper) or val_in_range(dist_ep2,self.dist_th_lower,self.dist_th_upper)):
                    # we have a connection! use the endpoint that is further away to find another object thats close to it

                    if (dist_ep1 <= dist_ep2):
                        # use endpoint 2
                        endpoint = line['endpoints'][1]
                        #print('{} at {} is close to {} with a distance of {}, using {} to detect a connection'.format(t1['name'], center, dist_ep1, line['endpoints'][0], line['endpoints'][1]))
                    else:
                        endpoint = line['endpoints'][0]
                        #print('{} at {} is close to {} with a distance of {}, using {} to detect a connection'.format(t1['name'], center, dist_ep2, line['endpoints'][1], line['endpoints'][0]))

                    # first check if the opposite endpoint is closest to the source
                    dist_source = distance_pts((source_position_unscaled, endpoint))
                    if (val_in_range(dist_source, source_dist_thresh_lower, source_dist_thresh_upper)):
                        # connected to the source
                        t1['connectedToSource'] = True
                        #print('Item {} is connected to the source'.format(t1['label']))
                        used_lines.append(k)
                        break
                    #else:
                        #print('Distance from source to endpoint was {}'.format(dist_source))
                    # iterate over all tracked objects again to see if the end of this line is close enough to any other object
                    for j,t2 in enumerate(self._tracking):
                        #print('made it this FAR!!! {} {}'.format(t1['id'], t2['id']))#now this DOES print for whatever reason...
                        if (t1['id'] == t2['id']):
                            # don't attempt to find connections to yourself
                            continue
                        # also don't attempt a connection if these two are already connected
                        if (((t2['id'], t2['label']) in t1['connectedToPrimary'])  or ((t2['id'], t2['label']) in t1['connectedToSecondary']) ):
                            continue

                        # check if the slope between the two rxrs and that of the line are similar
                        center2 = self._unscale_point(t2['center_scaled'], frameSize)
                        #print('the distance between objects {} {} and {} {} is {}'.format(t1['label'],t1['id'], t2['label'],t2['id'], distance_pts((center, center2))))#now this line is printing again...
                        lineSlope = line['slope']
                        lineAngle = np.pi/2.0 + np.arctan(lineSlope)#compare angles instead of slopes; bounded space from 0 to pi
                        rxrSlope, intercept = line_from_endpoints((center, center2)) if center[1] > center2[1] else line_from_endpoints((center2, center))
                        rxrAngle = np.pi/2.0 + np.arctan(rxrSlope)
                        angleDiff = abs(lineAngle - rxrAngle)#at most pi
                        #print('line angle is {} deg, rxr angle is {} deg, and angle % difference is {}'.format(lineAngle * 180./np.pi, rxrAngle * 180./np.pi, angleDiff/np.pi))#print in degrees for legibility
                        #sys.stdout.flush()
                        angleThresh = np.pi/6.0
                        if angleDiff > angleThresh:
                            continue
                        dist2 = distance_pts((center2, endpoint))
                        if (val_in_range(dist2, self.dist_th_lower,self.dist_th_upper)):
                            # its a connection! list this one as a connection, then break out of this loop
                            # we can create directionality by having two lists
                            # figure out which one is further to the left by checking the which x coordinate is greater (counter-intuitive, but the camera view is flipped)
                            # if equal, use the y coordinate
                            if (center[0] > center2[0]) or (center[0] == center2[0] and center[1] < center2[1]):
                                # first point is the primary
                                t1['connectedToPrimary'].append((t2['id'], t2['label']))
                                t2['connectedToSecondary'].append((t1['id'], t1['label']))
                            else:
                                t2['connectedToPrimary'].append((t1['id'],t1['label']))
                                t1['connectedToSecondary'].append((t2['id'], t2['label']))

                            #print('{} is connected to {}'.format(t1['name'], t2['name'])) #debug message
                            #print('Item {} is connected to {}'.format(t1['label'], t2['label']))

                            # make sure that the line used to discern this connection is not used again
                            used_lines.append(k)
                            break
                        else:
                            pass
                            #print('dist from object {} was {}'.format(t2['name'], dist2))



    def _unscale_point(self,point,shape): #TODO: move these to utils.py
        return (point[0]*shape[1], point[1]* shape[0])
    def _unscale(self, array, shape):
        return (array * [shape[1], shape[0]]).astype(np.int32)

    async def decorate_frame(self, frame, name):
        smaller_frame = frame
        if name == 'line-segmentation':
            frame = self.lineDetector.threshold_background(frame)
            return  frame#cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if name != 'track':
            return  smaller_frame#cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for i,t in enumerate(self._tracking):
            center_pos = tuple(np.array(self._unscale_point(t['center_scaled'], frame.shape)).astype(np.int32))
            #print('the center position is {}'.format(center_pos))
            cv2.circle(frame,center_pos, 10, (0,0, 255), -1)#draw red dots at centers of each polygon
            #draw the inner and outer dist thresholds for linefinding
            cv2.circle(frame, center_pos, self.dist_th_lower, (0,255, 255), 2)#BGR for yellow
            cv2.circle(frame, center_pos, self.dist_th_upper, (255,255, 0), 2)#BGR for cyan
            cv2.putText(frame,
                        '{}: {}'.format(t['name'], t['observed']),
                        (0, 60 * (i+ 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255))

        # view lines, as the decorator for LineProcessor is not working
        if (self.lineDetector is not None):
            for i,line in enumerate(self.lineDetector.lines):
                endpoints = line['endpoints']
                # place dots at each existing endpoint. blue for 0, green for 1
                cv2.circle(frame, (endpoints[0][0],endpoints[0][1]), 3 , (255,0,0), -1)#BGR
                cv2.circle(frame, (endpoints[1][0],endpoints[1][1]), 3 , (0,255,0), -1)#BGR

        return  frame#cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def KF_predict(self, X_prior2, X_prior, P_prior, F, F_P, Q, u, B):
        '''**INPUT:**
        X_prior2: the final output state, X, at the second to most recent timestep
        X_prior: The final output state, X, at the previous timestep
        P_prior: The state covariance matrix from the previous timestep
        F: The state "transition" matrix  that is based off of the linear dynamic system
        Q: The process noise covariance matrix
        u: A variable not part of the state but that impacts the state X
        B: Takes the input vector and puts into format that can be applied

        **OUTPUT:**
        X_intermediate: the initially predicted state of X
        P_intermediate: the initially predicted covariance matrix of the state, X
        NOTE: The above two output variables must be passed through KF_update_step to have a decently accurate
        prediction of the true value'''

        #Note: the X below is just the *initially* predicted state, which still needs to go through the update step
        X_prior = np.matrix([[X_prior[0,0]], [X_prior2[0,0]]])
        X_intermediate = dot(F, X_prior) + dot(B, u)
        P_intermediate = dot(F_P, dot(P_prior, F_P.T)) + Q
        return (X_intermediate, P_intermediate)

    def KF_update(self, X_intermediate, P_intermediate, Z, H, R):
        '''**INPUT:**
        X_intermediate: The *initially* predicted state based on previous state
        P_intermediate: The *initially* predicted covariance matrix of state X
        Z: The observed measurement
        H: Maps the observed measurement into the appropriate scale/units
        R: The measurement noise covariance matrix

        **LOCAL VARIABLES:**
        X_mean: the mean of the state, with corrected scale/units
        X_covar: the covariance of the state, with corrected scale/units
        K_gain: the Kalman Filter gain, which determines how much correction is needed

        **OUTPUT:**
        X_final: the final *corrected/updated* predicted state at current timestep
        P_final: the final *corrected/updated* state covariance matrix
        (Note that these essentially become X_prior and P_prior when inputed into KF_predict_step for the next timestep.)
        '''

        X_mean = dot(H, X_intermediate)
        X_covar = dot(H, dot(P_intermediate, H.T))

        K_gain = dot(P_intermediate, dot(H.T, inverse(X_covar + R)))

        X_final = X_intermediate + dot(K_gain, (Z - X_mean))
        P_final = P_intermediate - dot(K_gain, dot(H, K_gain.T))
        return (X_final, P_final)

    def track(self, frame, brect, poly, label, id_num, temperature = 300, volume = 200):
        '''
        Track a newly found object (returns True), or return False for an existing object.
        '''

        center = poly_scaled_center(poly, frame) if (poly is not None and cv2.contourArea(poly) < rect_area(brect)) else rect_scaled_center(brect, frame)
        # get current value of temperature - this will not change after the initialization
        if (self.dialReader is not None):
            temperature = self.dialReader.temperature
            volume = self.dialReader.volume
        else:
            temperature = 300
            volume = 200
        #we need to make sure we don't have an existing object here
        i=0

        #CONSTANT Kalman Filter vars:
        identity_matrix = np.matrix([1])
        u = np.matrix([0])
        B = identity_matrix #We are not considering u, so with that in mind, this was simply set as the identity matrix
        H = identity_matrix #The scale/units of the state are as we want them, so this is simply the identity matrix
        dampening_const = 1.0 #range: (0.0, 1.0]. Most likely it would be within (0.90, 1.00]
        F = np.matrix([(1+dampening_const), (-1*dampening_const)])
        F_P = np.matrix([1])
        Q = np.matrix([0])
        R_x = np.matrix([0.008256807930521852])
        R_y = np.matrix([0.009095864372886798])

        for t in self._tracking:
            if  t['name'] == '{}-{}'.format(label, id_num) or intersecting_rects(t['brect'], brect): #found already existing reactor
                t['observed'] = self.ticks_per_obs
                # t['center_scaled'] = [t['center_scaled'][0] * (1.0 - self.alpha) + center[0] * self.alpha, t['center_scaled'][1] * (1.0 - self .alpha) + center[1] * self.alpha] #do exponential averaging of position to cut down jitters

                if t['state_prior_x'] is None:
                    t['state_P_prior_x'], t['state_P_prior_y'] = identity_matrix, identity_matrix
                    t['state_prior2_x'], t['state_prior2_y'] = np.matrix([center[0]]), np.matrix([center[1]])
                    t['state_prior_x'], t['state_prior_y'] = np.matrix([center[0]]), np.matrix([center[1]])
                else:
                    t['state_P_prior_x'], t['state_P_prior_y'] = t['state_P_final_x'], t['state_P_final_y']
                    t['state_prior2_x'], t['state_prior2_y'] = t['state_prior_x'], t['state_prior_y']
                    t['state_prior_x'], t['state_prior_y'] = t['state_final_x'], t['state_final_y']

                #print('X_prior: {}'.format(t['state_prior_x']))

                t['state_intermed_x'], t['state_P_intermed_x'] = self.KF_predict(t['state_prior2_x'], t['state_prior_x'], t['state_P_prior_x'], F, F_P, Q, u, B)
                t['state_intermed_y'], t['state_P_intermed_y'] = self.KF_predict(t['state_prior2_y'], t['state_prior_y'], t['state_P_prior_y'], F, F_P, Q, u, B)

                Z_x = center[0]
                Z_y = center[1]

                t['state_final_x'], t['state_P_final_x'] = self.KF_update(t['state_intermed_x'], t['state_P_intermed_x'], Z_x, H, R_x)
                t['state_final_y'], t['state_P_final_y'] = self.KF_update(t['state_intermed_y'], t['state_P_intermed_y'], Z_y, H, R_y)
                t['center_scaled'] = [t['state_final_x'][0,0], t['state_final_y'][0,0]]

                # print('time: {} at observed_center: {} & center_scaled: {}'.format(time.time(), center, t['center_scaled']))

                t['brect'] = brect
                return False


        name = '{}-{}'.format(label, id_num)
        #tracker = cv2.DualTVL1OpticalFlow_create()
        #status = tracker.init(cv2.UMat(frame), brect)

        #if not status:
        #    print('Failed to initialize tracker')
        #    return False



        track_obj = {'name': name,
                     #'tracker': tracker,
                     'label': label,
                     'poly': poly,
                     'init': brect,
                     'area_init':rect_area(brect),
                     'center_scaled': poly_scaled_center(poly, frame) if (poly is not None and cv2.contourArea(poly) < rect_area(brect)) else rect_scaled_center(brect, frame),
                     'brect': brect,
                     'observed': self.ticks_per_obs,
                     'start': self.ticks,
                     'delta': np.int32([0,0]),
                     'id': id_num,
                     'connectedToPrimary': [],
                     'weight':[temperature,volume],#NOTE: this is where the actual values change!
                    #  'time_start': None,
                    #  'time_tot': None,
                     'state_prior2_x': None,
                     'state_prior2_y': None,
                     'state_prior_x': None,
                     'state_prior_y': None,
                     'state_intermed_x': None,
                     'state_intermed_y': None,
                     'state_P_prior_x': None,
                     'state_P_prior_y': None,
                     'state_P_intermed_x': None,
                     'state_P_intermed_y': None,
                     'state_final_x': None,
                     'state_final_y': None,
                     'state_P_final_x': None,
                     'state_P_final_y': None}
        self._tracking.append(track_obj)
        return True