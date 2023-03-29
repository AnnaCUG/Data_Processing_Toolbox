import os
import csv
import numpy as np
import cv2
from pycocotools.coco import COCO

image_path = "D:/spacenet/val/cut_image/"
imagefiles = os.listdir(image_path)
masksave = "D:/boundary-optimized/DSAC-master/models/spacenet/mask/"
data_path = 'D:/boundary-optimized/DSAC-master/models/spacenet/results/'
csvfile=open(data_path+'polygons.csv', newline='')
reader = csv.reader(csvfile)
coco = COCO("D:/spacenet/val/cut.json")
for file in imagefiles[:1000]:
    corners = reader.__next__()
    num_points = np.int32(corners[0])
    poly = np.zeros([num_points, 2])
    for c in range(num_points):
        poly[c, 0] = np.float(corners[1+2*c])
        poly[c, 1] = np.float(corners[2+2*c])
    poly[:,[1,0]]=poly[:,[0,1]]
    poly = poly.flatten().tolist()
    anns = coco.loadAnns(0)[0]
    anns['segmentation'] = [poly]
    mask = coco.annToMask(anns)
    cv2.imwrite(masksave + file, mask*255)
