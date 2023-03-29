import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

image_path = "D:/spacenet/val/cut_image/"
imagefiles = os.listdir(image_path)
imagesave = "D:/boundary-optimized/DSAC-master/models/spacenet/show/"
data_path = 'D:/boundary-optimized/DSAC-master/models/spacenet/results/'
csvfile=open(data_path+'polygons.csv', newline='')
reader = csv.reader(csvfile)
for file in imagefiles[:1000]:
    corners = reader.__next__()
    num_points = np.int32(corners[0])
    poly = np.zeros([num_points, 2])
    for c in range(num_points):
        poly[c, 0] = np.float(corners[1+2*c])
        poly[c, 1] = np.float(corners[2+2*c])
    poly[:,[1,0]]=poly[:,[0,1]]
    image = plt.imread(image_path + file)*255
    height, width = image.shape[:2]
    _, ax = plt.subplots(1, figsize=(16, 16))
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    p = Polygon(poly,facecolor='none',linewidth=4.5, edgecolor=(0,1,0))
    ax.add_patch(p)
    # for point in poly:
    #     ax.plot(point[0], point[1], color=[0, 1, 0], linewidth=2.0, marker='*')
    ax.imshow(image.astype(np.uint8))
    plt.savefig(imagesave + file)
    plt.close()
