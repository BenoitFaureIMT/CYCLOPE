import numpy as np
import cv2


from models.experimental import attempt_load
from utils.general import non_max_suppression
import torch

class YOLOv7(object):
    def __init__(self):
        self.device_choice = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_choice)
        #Load model
        self.model = attempt_load("YoloV7x-m-c.pt", map_location=self.device)
        #FP16 (seems to be if device != cpu)
        self.half = self.device_choice != 'cpu'
        if self.half:
            self.model.half()

        #Non max
        self.conf_thres = 0.25
        self.iou_thres = 0.45
    
    def warm_up(self):
        #Warmup
        if self.half:
            self.model(torch.zeros(1, 3, 640, 640).to(self.device).type_as(next(self.model.parameters()))) #TODO 640 to variable image size

    def run_net(self, img):

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0) #WTF?

        pred = self.model(img)[0] #What about augment, I dont know...
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres) #classes=opt.classes, agnostic=opt.agnostic_nms

        return np.array(pred[0].cpu())

if False:
    def display_yolo(out, img):
        for o in out:
            x,y,x2,y2 = o[:4]
            cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0,255,0), 1)
        
        cv2.imshow('Yolov5', img)

    det = YOLOv7()

    cam = cv2.VideoCapture("testfish.avi")
    ret = True
    while ret:
        print("-------------------")
        ret, frame = cam.read()
        frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
        disp = frame.copy()
        frame = frame[:, :, ::-1].transpose(2, 0, 1)
        frame = np.ascontiguousarray(frame)

        out = det.run_net(frame)
        print(out)
        
        display_yolo(out, disp)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()