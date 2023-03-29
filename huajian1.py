import json
import os
from math import sqrt, pow
import random
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def point2LineDistance(point_a, point_b, point_c):
    """
    计算点a到点b c所在直线的距离
    :param point_a:
    :param point_b:
    :param point_c:
    :return:
    """
    # 首先计算b c 所在直线的斜率和截距
    if point_b[0] == point_c[0]:
        return 9999999
    slope = (point_b[1] - point_c[1]) / (point_b[0] - point_c[0])
    intercept = point_b[1] - slope * point_b[0]

    # 计算点a到b c所在直线的距离
    distance = abs(slope * point_a[0] - point_a[1] + intercept) / sqrt(1 + pow(slope, 2))
    return distance


class DouglasPeuker(object):
    def __init__(self,threshold):
        self.threshold = threshold
        self.qualify_list = list()
        self.disqualify_list = list()

    def diluting(self, point_list):
        """
        抽稀
        :param point_list:二维点列表
        :return:
        """
        if len(point_list) < 3:
            self.qualify_list.extend(point_list[::-1])
        else:
            # 找到与收尾两点连线距离最大的点
            max_distance_index, max_distance = 0, 0
            for index, point in enumerate(point_list):
                if index in [0, len(point_list) - 1]:
                    continue
                distance = point2LineDistance(point, point_list[0], point_list[-1])
                if distance > max_distance:
                    max_distance_index = index
                    max_distance = distance

            # 若最大距离小于阈值，则去掉所有中间点。 反之，则将曲线按最大距离点分割
            if max_distance < self.threshold:
                self.qualify_list.append(point_list[-1])
                self.qualify_list.append(point_list[0])
            else:
                # 将曲线按最大距离的点分割成两段
                sequence_a = point_list[:max_distance_index]
                sequence_b = point_list[max_distance_index:]

                for sequence in [sequence_a, sequence_b]:
                    if len(sequence) < 3 and sequence == sequence_b:
                        self.qualify_list.extend(sequence[::-1])
                    else:
                        self.disqualify_list.append(sequence)

    def main(self, point_list):
        self.diluting(point_list)
        while len(self.disqualify_list) > 0:
            self.diluting(self.disqualify_list.pop())
        return self.qualify_list

class LimitVerticalDistance(object):
    def __init__(self,threshold):
        self.threshold = threshold
        self.qualify_list = list()

    def diluting(self, point_list):
        """
        抽稀
        :param point_list:二维点列表
        :return:
        """
        self.qualify_list.append(point_list[0])
        check_index = 1
        while check_index < len(point_list) - 1:
            distance = point2LineDistance(point_list[check_index],
                                          self.qualify_list[-1],
                                          point_list[check_index + 1])

            if distance < self.threshold:
                check_index += 1
            else:
                self.qualify_list.append(point_list[check_index])
                check_index += 1
        return self.qualify_list

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

    image = plt.imread("E:/WUH/cut/test2014/"+name[:-5]+".jpg")

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')

    for i, buliding in enumerate(annotations):
        color = colors[i]
        segmentation = buliding["segmentation"]
        for polygon in segmentation:
            polygon = np.fliplr(polygon) - 1
            p = Polygon(polygon, facecolor=color+(0.3,), edgecolor=color)
            ax.add_patch(p)
    ax.imshow(image.astype(np.uint8))
    plt.savefig("E:/boundary_results_WHU/Mask_RCNN_VD/image_results/" + os.path.splitext(name)[0] + ".jpg")
    plt.close()

indir = "E:/boundary_results_WHU/Mask_RCNN/json_results/"
outdir = "E:/boundary_results_WHU/Mask_RCNN_VD/json_results/"
files = os.listdir(indir)
for file in files:
    with open(indir+file, 'r') as load_json:
        json_dict = json.load(load_json)
    annotations = json_dict["annotations"]
    for i, buliding in enumerate(annotations):
        segmentation = buliding["segmentation"]
        for j, polygon in enumerate(segmentation):
            d = LimitVerticalDistance(0.3)
            p_d = d.diluting(polygon)
            json_dict["annotations"][i]["segmentation"][j] = p_d
    with open(outdir+file, 'w') as write_json:
        json.dump(annotations, write_json)
    write_json.close()
    display_instances(file, annotations)
    print(file)