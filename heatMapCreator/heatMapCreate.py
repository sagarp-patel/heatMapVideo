# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:40:27 2020

@author: Sagar
"""
import cv2
import numpy as np
import os
import re
import copy
from progress.bar import Bar

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)',text)]

def create_video(image_folder, name):
    images = [img for img in os.listdir(image_folder)]
    images.sort(key=natural_keys)

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height,width,layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    video = cv2.VideoWriter(name, fourcc, 30.0, (width, height))
    bar = Bar('Creating Video', max=len(images))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder,image)))
        bar.next()
    cv2.destroyAllWindows()
    video.release()

    for file in os.listdir(image_folder):
        os.remove(image_folder+file)
def create_heat_map_video(filename):
    capture = cv2.VideoCapture(filename)
    background_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    bar = Bar("Processing Frames",max=length)
    first_iteration_indicator = 1
    ac_img = 0
    
    for i in range(0,length):
        ret, frame = capture.read()
        if first_iteration_indicator == 1:
            first_frame = copy.deepcopy(frame)
            height,width = frame.shape[:2]
            accum_image = np.zeros((height,width),np.uint8)
            ac_img = accum_image
            first_iteration_indicator = 0
        else:
            filter = background_subtractor.apply(frame)
            threshold = 2
            maxValue = 2
            ret, th1 = cv2.threshold(filter, threshold, maxValue, cv2.THRESH_BINARY)
            accum_image = cv2.add(accum_image, th1)
            color_image_video = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
            video_frame = cv2.addWeighted(frame, 0.7, color_image_video, 0.7,0)
            name = "./frames/frame%d.jpg"%i
            cv2.imwrite(name,video_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        bar.next()
    bar.finish()
    create_video('./frames/','./output.avi')
    color_image = cv2.cvtColor(ac_img, cv2.COLORMAP_HOT)
    result_overlay = cv2.addWeighted(first_frame,0.7,color_image,0.7,0)
    cv2.immwrite('diff-overlay.jpg',result_overlay)
   
    capture.release()
    cv2.destroyAllWindows()
    
create_heat_map_video('input_3.mp4')
