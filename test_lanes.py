from infer import *
import cv2
import time
import numpy as np

"""Test trained lane detection model on self recorded video"""

video = cv2.VideoCapture("../Downloads/driving.avi")
infer_times = []

frame=0
frame_limit = -1
frames_per_sec = 5
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

while True:
    ret, img = video.read()
    if ret:
        if frame%frames_per_sec==0:
            # Preprocess the image: Change size and convert to tensor
            tic1 = time.time_ns()
            img_tensor = preprocess(img)
            tic2 = time.time_ns()

            # Infer lanes
            predict = lane_model.predict(img_tensor)

            # Display lanes
            tic3 = time.time_ns()
            video_maker(predict,img,frame,total_frames)
            toc = time.time_ns()

            # print("######## COMPUTE TIMES ############")
            # print("Preprocess: {} s\n".format((tic2-tic1)*10**-9))
            # print("Infer: {} s\n".format((tic3-tic2)*10**-9))
            # print("Visualize: {} s\n".format((toc-tic3)*10**-9))
            # print("Total w/o visualize: {} s\n".format((tic3-tic1)*10**-9))
            # print("Total: {} s\n".format((toc-tic1)*10**-9))
            infer_times.append((tic3-tic1)*10**-9)
        frame+=1
    else:
        print("Average inference time for lane detection: {} s".format(np.average(np.array(infer_times))))
        break

    if frame==frame_limit:
        out.release()
        print("Average inference time for lane detection: {} s".format(np.average(np.array(infer_times))))
        break