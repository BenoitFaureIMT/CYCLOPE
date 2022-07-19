import cv2
import numpy as np
from Detection import YOLOv5
from Tracker import Cyclop

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

def display_both(targs, out, img):
    for o in out:
        y,x,y2,x2 = o[:4]*640
        cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0,255,0), 1)
    
    for t in targs:
        x,y,w,h = t.state[0]*640,t.state[1]*640,t.state[2]*640,t.state[3]*640
        cv2.rectangle(img, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (255,0,0), 1)
    
    cv2.imshow('Yolov5 + FishSORT', img)

tr = Cyclop()
det = YOLOv5("saved_model")
det.warm_up()

cam = cv2.VideoCapture("testfish.avi")
ret = True
while ret:
    print("-------------------")
    ret, frame = cam.read()
    frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
    out = det.process_output(det.run_net(frame))
    print("Detections : ", len(out))

    #Preprocess detection
    # out_p = np.zeros(out.shape)
    # out_p[:, 2], out_p[:, 3] = out[:, 3] - out[:, 1], out[:, 2] - out[:, 0]
    # out_p[:, 0], out_p[:, 1] = out[:, 1] + out_p[:, 3]/2, out[:, 0] + out_p[:, 2]/2

    tr.update(out.copy(), frame, 0.1)

    #print(len(tr.targs))
    #if len(tr.targs) > 0: 
    #print(tr.targs[0].state)
    #display(tr.targs, frame)
    #display_yolo(out, frame)
    display_both(tr.targs, out, frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()