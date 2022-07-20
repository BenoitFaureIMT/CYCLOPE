import tensorflow as tf
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

class ResNet50(object):
    def __init__(self, weights = "imagenet", input_shape = (224, 224), pooling = "avg"):
        self.model = tf.keras.applications.ResNet50(
            include_top=False,
            weights=weights,
            input_shape=input_shape + (3,),
            pooling=pooling)
        self.input_shape = input_shape
    
    def extract_sub_image(self, img, bbox): #bbox -> [y1,x1,y2,x2] (y1, x1) top left, (y2, x2) bottom right
        h, w, c = img.shape
        return img[int(bbox[0]*h):int(bbox[2]*h), int(bbox[1]*w):int(bbox[3]*w)]
        # h, w, c = img.shape
        # x, y, x2, y2 = bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2, bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2 #TODO change this shit
        # print(x*w, y*h, x2*w, y2*h)
        # return img[int(y*h):int(y2*h),int(x*w):int(x2*w)]
    
    def extract_sub_image_v2(self, img, bbox):#bbox -> [x, y, w, h]
        h, w, c = img.shape
        return img[int(bbox[1]*h):int((bbox[1] + bbox[3])*h),int(bbox[0]*w):int((bbox[0] + bbox[2])*w)]

    def extract_features(self, img):
        img = cv2.resize(img, self.input_shape, cv2.INTER_LINEAR) / 255
        img = np.array([img]) #Be carefull here...
        f = self.model.predict(img)
        return (f / np.linalg.norm(f))[0]
    
    def get_distance(self, t1, t2, dist = "EuclideanDistance"):
        return self.distances[dist](self, t1, t2)
    
    def get_features(self, img, bbox):
        return self.extract_features(self.extract_sub_image(img, bbox))
    
    def extract_cost_matrix(self, targets, detection_features):
        cost_matrix = np.zeros((len(targets), len(detection_features)))
        for i in range(cost_matrix.shape[0]):
            for j in range(cost_matrix.shape[1]):
                cost_matrix[i, j] = min([self.get_distance(tFeature, detection_features[j]) for tFeature in targets[i].features]) #d(i, j) = min{dist(rj, ri_k) | ri_k in Ri}
        return cost_matrix
    
    def get_cost_matrix(self, targets, img, bboxs):
        detection_features = np.array([self.get_features(img, bbox) for bbox in bboxs])
        return self.extract_cost_matrix(targets, detection_features), detection_features
    
    def extract_associations(self, cost_matrix):
        return np.argmin(cost_matrix, axis = 1)

    def euclidean_distance(self, t1, t2):
        return np.linalg.norm(t1 - t2)
    
    distances = {
        "EuclideanDistance" : euclidean_distance
    }


class ResNeXt50(object):
    def __init__(self, input_shape = (224, 224)):
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.input_shape = input_shape
    
    def extract_sub_image(self, img, bbox): #bbox -> [y1,x1,y2,x2] (y1, x1) top left, (y2, x2) bottom right
        h, w, c = img.shape
        return img[int(bbox[0]*h):int(bbox[2]*h), int(bbox[1]*w):int(bbox[3]*w)]
        # h, w, c = img.shape
        # x, y, x2, y2 = bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2, bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2 #TODO change this shit
        # print(x*w, y*h, x2*w, y2*h)
        # return img[int(y*h):int(y2*h),int(x*w):int(x2*w)]
    
    def extract_sub_image_v2(self, img, bbox):#bbox -> [x, y, w, h]
        h, w, c = img.shape
        return img[int(bbox[1]*h):int((bbox[1] + bbox[3])*h),int(bbox[0]*w):int((bbox[0] + bbox[2])*w)]

    def extract_features(self, img):
        input_image = Image.open(img)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_batch).flatten(start_dim=1)
        return np.array((output / np.linalg.norm(output))[0])
    
    def get_distance(self, t1, t2, dist = "EuclideanDistance"):
        return self.distances[dist](self, t1, t2)
    
    def get_features(self, img, bbox):
        return self.extract_features(self.extract_sub_image(img, bbox))
    
    def extract_cost_matrix(self, targets, detection_features):
        cost_matrix = np.zeros((len(targets), len(detection_features)))
        for i in range(cost_matrix.shape[0]):
            for j in range(cost_matrix.shape[1]):
                cost_matrix[i, j] = min([self.get_distance(tFeature, detection_features[j]) for tFeature in targets[i].features]) #d(i, j) = min{dist(rj, ri_k) | ri_k in Ri}
        return cost_matrix
    
    def get_cost_matrix(self, targets, img, bboxs):
        detection_features = np.array([self.get_features(img, bbox) for bbox in bboxs])
        return self.extract_cost_matrix(targets, detection_features), detection_features
    
    def extract_associations(self, cost_matrix):
        return np.argmin(cost_matrix, axis = 1)

    def euclidean_distance(self, t1, t2):
        return np.linalg.norm(t1 - t2)
    
    distances = {
        "EuclideanDistance" : euclidean_distance
    }