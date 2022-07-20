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

def draw_txt(img, label, x1, y1, color = (0, 0, 255), text_color = (255, 255, 255)):
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

    cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

def display_both(targs, out, ind_dets, img):
    for i in range(len(out)):
        y,x,y2,x2 = out[i][:4]*640
        cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0,255,0), 1)
        draw_txt(img, str(ind_dets[i]), int(x), int(y), color = (0,255,0), text_color = (0,0,255))
    
    for t in targs:
        x,y,w,h = t.state[0]*640,t.state[1]*640,t.state[2]*640,t.state[3]*640
        cv2.rectangle(img, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0,0,255) if t.missed_detection else (255, 0, 0), 1)
        draw_txt(img, str(t.id), int(x - w/2), int(y + h/2), color = (255, 0, 0))
    
    cv2.imshow('Yolov5 + FishSORT', img)

tr = Cyclop()
det = YOLOv5("saved_model")
det.warm_up()

init_img = np.ones((640, 640, 3)) * 255
cv2.putText(init_img, "PRESS SPACE TO START", (320 - 160, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
cv2.imshow('Yolov5 + FishSORT', init_img)
while True:
    k = cv2.waitKey(100)
    if k == 32:
        break

cam = cv2.VideoCapture("testfish.avi")
ret = True
while ret:
    print("-------------------")
    ret, frame = cam.read()
    frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
    out = det.process_output(det.run_net(frame))
    print("Detections : ", len(out))

    ind_dets = tr.update(out.copy(), frame, 0.01)

    display_both(tr.targs, out, ind_dets, frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()