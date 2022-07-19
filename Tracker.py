#Dependencies
import numpy as np
from scipy.spatial.distance import cdist
import lap

from ReID import ResNet50
from Filter import NNFilter
from Target import target
from Utils import IoU

#Class containing the tracker logic
class Cyclop(object):
    def __init__(self, filter_weight_path = "filter_weights.npy", reid = None, kalman = None, association = None): #screen_w, screen_h,
        #Initialise ReID
        self.reid = ResNet50() if reid == None else reid
        
        #Initialise Filter
        self.filter = NNFilter(filter_weight_path)

        #Initialise targets
        self.targs = np.array([])
        self.age_max = 10

        #Initialise EMA
        self.alpha = 0.9

    #Update the state of the tracker
    def update(self, detections, image, dt):

        #Coefficients for filtering TODO : change value and location of variables
        IoU_threshold = 0.2
        cosine_threshold = 0.2
        cost_threshold = 0.2

        #Get new state predictions for each target
        for t in self.targs:
            self.filter.pred_next_state(t, dt)

        #Obtain features
        detection_features = np.array([self.reid.get_features(image, bbox) for bbox in detections]) #TODO : make sure this works

        #Change detections format
        detections[:, 2], detections[:, 3] = detections[:, 3] - detections[:, 1], detections[:, 2] - detections[:, 0]
        detections[:, 0], detections[:, 1] = detections[:, 1] + detections[:, 2]/2, detections[:, 0] + detections[:, 3]/2
        
        #Cost matrix calculation
        cost_matrix = None
        match, unm_tr, unm_det = None, None, None
        if (len(self.targs) == 0 or len(detections) == 0):
            cost_matrix = np.array([[]])
            match, unm_tr, unm_det = [], range(len(self.targs)), range(len(detections))
        else:
            #   Calculate IOU cost matrix
            cost_matrix_IoU = np.array([[1 - IoU(t.pred_state.T[0], d) for d in detections] for t in self.targs]) #TODO : calculate it...
            cost_matrix_IoU[cost_matrix_IoU > IoU_threshold] = 1.0

            #   Calculate features cost matrix
            cost_matrix_feature = np.maximum(0.0, cdist(np.array([t.features for t in self.targs]), detection_features, metric='cosine')) / 2.0 #TODO : better way of creating array? Maybe store separately?
            cost_matrix_feature[cost_matrix_feature > cosine_threshold] = 1.0

            #   Final cost matrix calculation
            cost_matrix = np.minimum(cost_matrix_IoU, cost_matrix_feature)
            #Get associations
            match, unm_tr, unm_det = self.associate(cost_matrix, cost_threshold)

        print(match, unm_tr, unm_det)

        #Process associations
        #   Targets which were matched
        new_targs = []
        for ind_track, ind_det in match:
            targ = self.targs[ind_track]
            self.filter.update_state(targ, detections[ind_det], dt)
            targ.update_feature(detection_features[ind_det], self.alpha)
            new_targs.append(targ)

        #   Targets which were not matched
        for ind_unm_tr in unm_tr:
            targ = self.targs[ind_unm_tr]
            self.filter.update_state_no_detection(targ, dt)
            if targ.age <= self.age_max:
                new_targs.append(targ)
        
        #   New targets
        for ind_unm_det in unm_det:
            new_targs.append(target(detections[ind_unm_det], detection_features[ind_unm_det]))
        
        self.targs = np.array(new_targs)
    
    def associate(self, cost_mat, cost_thres): # TODO : does this work?
        if cost_mat.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_mat.shape[0])), tuple(range(cost_mat.shape[1]))
        matches, unmatch_track, unmatch_detection = [], [], []
        __, x, y = lap.lapjv(cost_mat, extend_cost=True, cost_limit=cost_thres)
        for ix, mx in enumerate(x):
            if mx >= 0:
                matches.append([ix, mx])
        unmatch_track = np.where(x < 0)[0]
        unmatch_detection = np.where(y < 0)[0]
        matches = np.asarray(matches)
        return matches, unmatch_track, unmatch_detection