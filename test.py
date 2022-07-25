import cv2
from matplotlib.pyplot import draw
import numpy as np
import time

import torch

from Detection import YOLOv7
from Tracker import Cyclop
from ReID import ResNeXt50

def display(targs, img):
    for t in targs:
        y,x,h,wh = t.state[0]*640,t.state[1]*640,t.state[2]*640,t.state[3]
        w = wh*h
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255,0,0), 1)

    cv2.imshow('BBox', img)

def display_yolo(out, img):
    for o in out:
        y,x,y2,x2 = o[:4]*640
        cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0,255,0), 1)
    
    cv2.imshow('Yolov5', img)

def draw_txt(img, label, x1, y1, color = (0, 0, 255), text_color = (255, 255, 255)):
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

    cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

def display_both(targs, out, ind_dets, img):
    for i in range(len(out)):
        x,y,x2,y2 = out[i][:4]
        cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0,255,0), 1)
        #draw_txt(img, str(int(ind_dets[i])) + " | " + str(int(out[i][4] * 100)) + "%", int(x), int(y), color = (0,255,0), text_color = (0,0,255))
    
    for t in targs:
        x,y,w,h = t.state[0]*640,t.state[1]*640,t.state[2]*640,t.state[3]*640
        cv2.rectangle(img, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0,0,255) if t.missed_detection else (255, 0, 0), 1)
        draw_txt(img, str(t.id), int(x - w/2), int(y + h/2), color = (255, 0, 0))
    
    cv2.imshow('Yolov5 + FishSORT', img)

det = YOLOv7()
det.warm_up()
tr = Cyclop(reid = ResNeXt50(det.device), filter_weight_path = "filter_weights.npy")

if False:
    init_img = np.ones((640, 640, 3)) * 255
    cv2.putText(init_img, "PRESS SPACE TO START", (320 - 160, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow('Yolov5 + FishSORT', init_img)
    while True:
        k = cv2.waitKey(100)
        if k == 32:
            break

cam = cv2.VideoCapture("testfish.avi") # testfish.avi | MW-18Mar-1.avi
ret, frame = cam.read()
while ret:
    print("-------------------")
    t = time.perf_counter()

    frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
    disp = frame.copy()
    frame = frame[:, :, ::-1].transpose(2, 0, 1)
    frame = np.ascontiguousarray(frame)

    out = det.run_net(frame)
    out = np.clip(out, a_min = 0, a_max = 640)
    #out = out[out[:,4] > 0.35]

    print("Detections : ", len(out))
    print("Detection : ", int((time.perf_counter() - t) * 1000), " ms")

    ind_dets = tr.update(out.copy(), disp, 0.01)
    
    print("Total : ", int((time.perf_counter() - t) * 1000), " ms")

    display_both(tr.targs, out, ind_dets, disp)
    k = cv2.waitKey(1)
    if k == 27:
        break

    #Read next
    ret, frame = cam.read()

cam.release()
cv2.destroyAllWindows()