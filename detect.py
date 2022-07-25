import argparse

import cv2
import numpy as np
import time
import torch

from Detection import YOLOv7
from Tracker import Cyclop
from ReID import ResNeXt50

def draw_txt(img, label, x1, y1, color = (0, 0, 255), text_color = (255, 255, 255)):
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

    cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

def display_both(targs, out, ind_dets, img):
    img_h = img.shape[0]
    img_w = img.shape[1]

    for i in range(len(out)):
        x,y,x2,y2 = out[i][:4]
        cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0,255,0), 1)
        #draw_txt(img, str(int(ind_dets[i])) + " | " + str(int(out[i][4] * 100)) + "%", int(x), int(y), color = (0,255,0), text_color = (0,0,255))
    
    for t in targs:
        x,y,w,h = t.state[0]*img_w,t.state[1]*img_h,t.state[2]*img_w,t.state[3]*img_h
        cv2.rectangle(img, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0,0,255) if t.missed_detection else (255, 0, 0), 1)
        draw_txt(img, str(t.id), int(x - w/2), int(y + h/2), color = (255, 0, 0))
    
    return img

@torch.no_grad()
def run(
    wait_screen = False,
    show_results = True,
    save_results = False,
    save_MOT = False,

    video = "testfish.avi",

    weights = "YoloV7x-m-c.pt",

    filter_weight_path = "filter_weights.npy", 
    age_max = 6, 
    alpha = 0.6, 
    IoU_threshold = 0.4, 
    cosine_threshold = 0.25, 
    cost_threshold = 0.4):

    det = YOLOv7(weights = weights)
    det.warm_up()
    tr = Cyclop(
        reid = ResNeXt50(det.device), 
        filter_weight_path = filter_weight_path, 
        age_max = age_max,
        alpha = alpha,
        IoU_threshold = IoU_threshold,
        cosine_threshold = cosine_threshold,
        cost_threshold = cost_threshold)

    if wait_screen:
        init_img = np.ones((640, 640, 3)) * 255
        cv2.putText(init_img, "PRESS SPACE TO START", (320 - 160, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Yolov5 + FishSORT', init_img)
        while True:
            k = cv2.waitKey(100)
            if k == 32:
                break

    cam = cv2.VideoCapture(video) # testfish.avi | MW-18Mar-1.avi
    ret, frame = cam.read()

    if save_MOT:
        frame_count = 1
        mot_file = open(video.split('.')[0] + ".txt", 'w')

    interference = []

    if(save_results):
        writer = cv2.VideoWriter('output_' + video, cv2.VideoWriter_fourcc(*'DIVX'), 25, (640, 640))

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
        
        interference.append((time.perf_counter() - t) * 1000)
        print("Total : ", int(interference[-1]), " ms")

        if show_results or save_results:
            img = display_both(tr.targs, out, ind_dets, disp)
            if show_results:
                cv2.imshow('Yolov5 + FishSORT', img)
            if save_results:
                writer.write(img)

        k = cv2.waitKey(1)
        if k == 27:
            break

        #MOT
        if save_MOT:
            for t in tr.targs:
                mot_file.write(
                    str(frame_count) + ", " + str(t.id) + ", " + 
                    str(t.state[0][0] * 640) + ", " + str(t.state[1][0] * 640) + ", " + 
                    str(t.state[2][0] * 640) + ", " + str(t.state[3][0] * 640) + ", -1, -1, -1, -1\n")
            frame_count += 1

        #Read next
        ret, frame = cam.read()

    print("Average total interference : Detection + ReID + Tracking : ", int(sum(interference)/len(interference)), " ms")
    print("Max id detected : ", tr.next_id - 1)

    cam.release()
    cv2.destroyAllWindows()
    if save_MOT:
        mot_file.close()
    if save_results:
        writer.release()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait_screen", action='store_true', help = "add a wait screen before tracking starts")
    parser.add_argument("--show_results", action='store_true', help = "display results on screen")
    parser.add_argument("--save_results", action='store_true', help = "save display results")
    parser.add_argument("--save_MOT", action='store_true', help = "save targets to file in MOT format")
    parser.add_argument("--video", type = str, default = "testfish.avi", help = "path to video file")
    parser.add_argument("--weights", type = str, default = "YoloV7x-m-c.pt", help = "path to YoloV7 weights")
    parser.add_argument("--filter_weight_path", type = str, default = "filter_weights.npy", help = "path to weiths of NNFilter")
    parser.add_argument("--age_max", type = int, default = 6, help = "age max of tracklets")
    parser.add_argument("--alpha", type = float, default = 0.6, help = "alpha coeff for EMA")
    parser.add_argument("--IoU_threshold", type = float, default = 0.4, help = "IoU threshold for association")
    parser.add_argument("--cosine_threshold", type = float, default = 0.25, help = "Cosine threshold for association")
    parser.add_argument("--cost_threshold", type = float, default = 0.4, help = "Cosine threshold for association")
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)