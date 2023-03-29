import json
import os
from math import sqrt, pow
import random
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pycocotools.coco import COCO

# 原coco数据集的路径
dataDir = "D:/spacenet/val/cut_image/"
annFile = 'D:/spacenet/val/cut.json'
imagesave = "D:/spacenet/val/cut_show/"
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

def display_instances(name, annotations, figsize=(16, 16)):
    _, ax = plt.subplots(1, figsize=figsize)
    colors = random_colors(len(annotations))

    image = plt.imread(dataDir+os.path.splitext(name)[0]+".png")*255

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')

    for i, buliding in enumerate(annotations):
        #color = colors[i]
        segmentation = buliding["segmentation"]
        for polygon in segmentation:
            polygon = np.array(polygon).reshape([-1,2]) - 0.5
            p = Polygon(polygon,facecolor='none',linewidth=4.5, edgecolor=(0,1,0))
            ax.add_patch(p)
            # for point in polygon:
            #     ax.plot(point[0], point[1], color=[0, 1, 0], markersize=20.0, marker='*')
    ax.imshow(image.astype(np.uint8))
    plt.savefig(imagesave + os.path.splitext(name)[0] + ".jpg")
    plt.close()

if __name__ == '__main__':
    # 按单个数据集进行处理
    coco = COCO(annFile)
    info = coco.dataset['info']
    categories = coco.dataset['categories']
    images = []
    annotations = []
    id = 0
    # 拿到所有需要的图片数据的id - 我需要的类别的categories的id是多少
    classes_ids = coco.getCatIds(catNms=classes_names)
    # 取所有类别的并集的所有图片id
    # 如果想要交集，不需要循环，直接把所有类别作为参数输入，即可得到所有类别都包含的图片
    imgIds_list = []
    # 循环取出每个类别id对应的有哪些图片并获取图片的id号
    for idx in classes_ids:
        imgidx = coco.getImgIds(catIds=idx)  # 将该类别的所有图片id好放入到一个列表中
        imgIds_list += imgidx
        print("搜索id... ", imgidx)
    # 去除重复的图片
    imgIds_list = list(set(imgIds_list))  # 把多种类别对应的相同图片id合并

    # 一次性获取所有图像的信息
    image_info_list = coco.loadImgs(imgIds_list)

    # 对每张图片生成一个mask
    for imageinfo in image_info_list:
        # 获取对应类别的分割信息
        annIds = coco.getAnnIds(imgIds=imageinfo['id'], catIds=classes_ids, iscrowd=None)
        anns_list = coco.loadAnns(annIds)
        image = plt.imread(dataDir + imageinfo['file_name'])
        display_instances(imageinfo['file_name'],anns_list)