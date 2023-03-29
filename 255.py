import cv2
import os
images = os.listdir("D:/crowdAI/val/label/")
for i in range(60317):
    path_in = "D:/crowdAI/val/label/" + images[i]
    path_out = "D:/crowdAI/val/label_255/" + images[i]
    x = cv2.imread(path_in)
    x255 = x * 255
    cv2.imwrite(path_out, x255)