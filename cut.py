import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import numpy as np
from PIL import Image
import cv2
import random
path=r"C:/Users/chens/Desktop/Building change detection dataset/Building change detection dataset/1. The two-period image data/before/"
out=r"C:/Users/chens/Desktop/before256/"
file=r"before.tif"
std_size=256
step=256
#image = Image.open(path+file)
files=os.listdir(path)
#for file in files:
for i in range(1):
    if 1:
    #if file[-9:-4]=="image":
        print(path+file)
        image_np=cv2.imread(path+file,-1)
        h, w, c = image_np.shape
        # h, w, c = image_np.shape
        # labels = labels[:,:,2]
        # labels = 255 - labels
        # cv2.imwrite(path+file[:-9]+"labels_road.png",labels)
        for start_h in range(0, h-std_size+step, step):
            for start_w in range(0, w-std_size+step, step):
                if start_h + std_size > h:
                    start_h = h - std_size
                if start_w + std_size > w:
                    start_w = w - std_size
                image_cut=image_np[start_h:start_h+std_size,start_w:start_w+std_size]
                prefix = os.path.basename(file).strip('.jpg').strip('.tif').strip('.png')
                image_filename = "{0}_cut{1}{2}".format(file[:-4], str(start_h).zfill(5), str(start_w).zfill(5))
                cv2.imwrite(out+"/"+image_filename+".tif",image_cut) 
                pass
                # for r in range(4):
                #     image_r = np.rot90(image_cut,r)
                #     image_filename = "{0}_cut{1}{2}_r{3}".format(file[:-4], str(start_h).zfill(5), str(start_w).zfill(5), str(r*90).zfill(3))
                #     cv2.imwrite(out+"/"+image_filename+".tif",image_r) 
