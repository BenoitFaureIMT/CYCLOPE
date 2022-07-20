import cv2
from ReID import ResNeXt50

cam = cv2.VideoCapture("testfish.avi")
model = ResNeXt50()
ret = True
while ret:
    print("-------------------")
    ret, frame = cam.read()
    print("Read Frame")
    frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
    print("Resized Frame")
    print(model.get_features(frame, [30/640, 30/640, 120/640, 60/640]))

cam.release()