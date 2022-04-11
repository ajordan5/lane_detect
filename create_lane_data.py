import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torch
from tqdm import tqdm

"""Create labels for lane lines on the TuSimple Dataset"""

class laneTrain:
    def __init__(self, indices=[0], val_or_train='train') -> None:
        self.colors = [[255,0,0],[0,255,0],[0,0,255],[0,255,255], [255,0,255]]
        file_in = '/home/boatlanding/Downloads/train_set'
        self.file_out = '/home/boatlanding/datasets/tusimple/' + val_or_train
        self.img_files = []
        self.labels = dict()
        self.resize = (640,384)
        self.pad = np.zeros((24, 640, 3))

        # Load data and create labels
        self.load_json(file_in, indices)
        # np.savez(os.path.join(file_in, "labels.npz") ,self.labels)
        
    def load_json(self,file_in, indices):
        contents = [f for f in os.listdir(file_in) if f.endswith(".json")]
        for idx in indices:
            file = contents[idx]
            if file.endswith(".json"):
                json_file = os.path.join(file_in, file)
                json_gt = [json.loads(line) for line in open(json_file)]

                # Unpack lane markings for each clip and create its label
                for idx in tqdm(range(len(json_gt)), desc="Generating labels"):
                    gt = json_gt[idx]
                    gt_lanes = gt['lanes']
                    y_samples = gt['h_samples'] 
                    raw_file = gt['raw_file']
                    img_file = os.path.join(file_in, raw_file)
                    img = cv2.imread(img_file)
                    
                    
                    self.img_files.append(img_file)
                    if idx == 1430:
                        self.file_out = '/home/boatlanding/datasets/tusimple/val'

                    # Label creation
                    label = self.create_label(img, gt_lanes, y_samples)
                    label_file = img_file.split("/")[-2]+".jpg"
                    # label_file = "/".join(img_file.split("/")[:-1])

                    # Write out resized image and label
                    img = cv2.resize(img, self.resize)
                    label = cv2.resize(label, self.resize)
                    cv2.imwrite(os.path.join(self.file_out, "pic", label_file), img)
                    cv2.imwrite(os.path.join(self.file_out, "label", label_file), label)
                

    def create_label(self, img, lanes, y_samples):
        mask = np.zeros_like(img)
        gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples)
                  if x >= 0] for lane in lanes]
        for i in range(len(gt_lanes_vis)):
            cv2.polylines(mask, np.int32([gt_lanes_vis[i]]), isClosed=False,color=self.colors[i], thickness=5)
            
        # create grey-scale label image
        label = np.ones((720,1280),dtype = np.uint8)
        for i in range(len(self.colors)):
            label[np.where((mask == self.colors[i]).all(axis = 2))] = 0
        return label

if __name__ == "__main__":
    
    # TuSimple Train set has three folders, choose which folder to extract and whether the pictures/labels need to go into 'train' or 'val'
    
    trainer = laneTrain(indices=[2], val_or_train='train') 