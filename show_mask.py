import os
import random
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from skimage.measure import find_contours

# 原coco数据集的路径
maskdir = "D:/HAN/HRNet-Semantic-Segmentation-HRNet-OCR/test_results/"
imagedir = 'D:/HAN/spacenet/val2014/'
imagesave = "D:/HAN/HRNet-Semantic-Segmentation-HRNet-OCR/show_results/"
classes_names = ['building']

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def display_instances(name, figsize=(16, 16)):
    _, ax = plt.subplots(1, figsize=figsize)

    image = plt.imread(imagedir+os.path.splitext(name)[0]+".jpg")
    mask = plt.imread(maskdir+os.path.splitext(name)[0]+".png")[:,:,0]

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask 
    contours = find_contours(padded_mask, 0.5)
    colors = random_colors(len(contours))
    for i, verts in enumerate(contours):
        verts = np.fliplr(verts) - 0.5
        p = Polygon(verts, facecolor=colors[i]+(0.3,), edgecolor=colors[i])
        ax.add_patch(p)
    ax.imshow(image.astype(np.uint8))
    plt.savefig(imagesave + os.path.splitext(name)[0] + ".jpg")
    plt.close()

if __name__ == '__main__':
    files = os.listdir(imagedir)
    for file in files:
        display_instances(file)