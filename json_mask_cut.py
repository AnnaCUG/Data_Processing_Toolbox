"""
	需要修改的地方:
		dataDir,savepath改为自己的路径
		class_names改为自己需要的类
		dataset_list改为自己的数据集名称
"""
from __future__ import annotations
from pycocotools.coco import COCO
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
from shapely.affinity import translate
from shapely.ops import polygonize, unary_union
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString

'''
路径参数
'''
# 原coco数据集的路径
dataDir = "D:/spacenet/val2014/"
annFile = 'D:/boundary-optimized/Mask_RCNN_X_spacenet/out.json'
# 用于保存新生成的mask数据的路径
masksave = "D:/boundary-optimized/Mask_RCNN_X_spacenet/cut_mask/"
imagesave = "D:/boundary-optimized/Mask_RCNN_X_spacenet/cut_image/"
jsonsave = "D:/boundary-optimized/Mask_RCNN_X_spacenet/"

'''
数据集参数
'''
# coco有80类，这里写要进行二值化的类的名字
# 其他没写的会被当做背景变成黑色
classes_names = ['building']


# 生成保存路径，函数抄的(›´ω`‹ )
# if the dir is not exists,make it,else delete it
def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)


# 生成mask图
def mask_generator(coco, width, height, anns_list):
    mask_pic = np.zeros((height, width))
    # 生成mask - 此处生成的是4通道的mask图,如果使用要改成三通道,可以将下面的注释解除,或者在使用图片时搜相关的程序改为三通道
    for single in anns_list:
        mask_single = coco.annToMask(single)
        mask_pic += mask_single
    # 转化为255
    for row in range(height):
        for col in range(width):
            if (mask_pic[row][col] > 0):
                mask_pic[row][col] = 255
    mask_pic = mask_pic.astype(int)
    return mask_pic

    # 转为三通道
    # imgs = np.zeros(shape=(height, width, 3), dtype=np.float32)
    # imgs[:, :, 0] = mask_pic[:, :]
    # imgs[:, :, 1] = mask_pic[:, :]
    # imgs[:, :, 2] = mask_pic[:, :]
    # imgs = imgs.astype(np.uint8)
    # return imgs


# 处理json数据并保存二值mask
def get_mask_data(insize, outsize):
    # 获取COCO_json的数据
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
        # # 生成二值mask图
        # mask_image = mask_generator(coco, imageinfo['width'], imageinfo['height'], anns_list)
        # # 保存图片
        # file_name = mask_to_save + '/' + imageinfo['file_name'][:-4] + '.jpg'
        # plt.imsave(file_name, mask_image)
        # print("已保存mask图片: ", file_name)
        for i, anns in enumerate(anns_list):
            mask = coco.annToMask(anns)
            segmentation = anns['segmentation']
            # poly = np.round(np.array(segmentation[0]).reshape(-1, 2))
            poly = MultiPolygon([Polygon(np.array(polygon).reshape(-1, 2)) for polygon in segmentation])
            if not poly.is_valid:
                poly = []
                for polygon in segmentation:
                    p = np.array(polygon).reshape(-1, 2)
                    c = 0
                    for j,point in enumerate(p):
                        if all(point == p[c]) and j > c+2:
                            poly.append(Polygon(p[c:j+1]))
                            c = j+1
                poly = MultiPolygon(poly)
                if not poly.is_valid:
                    poly = MultiLineString([LineString(np.array(polygon).reshape(-1, 2)) for polygon in segmentation])
                    poly = unary_union(poly)
                    poly = MultiPolygon(list(polygonize(poly)))
            if not poly.is_valid or poly.area == 0:
                print("无效图形" + imageinfo['file_name'] + '——' + str(i))
                continue
            frame = Polygon([[0, 0], [0, insize], [insize, insize], [insize, 0], [0, 0]])
            poly = poly.intersection(frame)
            cx = poly.bounds[0] + (poly.bounds[2] - poly.bounds[0]) / 2
            cy = poly.bounds[1] + (poly.bounds[3] - poly.bounds[1]) / 2
            x = round(outsize / 2 - cx)
            y = round(outsize / 2 - cy)
            msk = np.zeros([outsize, outsize], dtype=np.uint8)
            img = np.zeros([outsize, outsize, 3], dtype=np.uint8)
            msk[y if y > 0 else 0: insize + y if y < outsize - insize else outsize,
            x if x > 0 else 0: insize + x if x < outsize - insize else outsize] = mask[
                                                                                  0 - y if y < 0 else 0: outsize - y if y > outsize - insize else insize,
                                                                                  0 - x if x < 0 else 0: outsize - x if x > outsize - insize else insize]
            img[y if y > 0 else 0: insize + y if y < outsize - insize else outsize,
            x if x > 0 else 0: insize + x if x < outsize - insize else outsize] = image[
                                                                                  0 - y if y < 0 else 0: outsize - y if y > outsize - insize else insize,
                                                                                  0 - x if x < 0 else 0: outsize - x if x > outsize - insize else insize]
            file_name = os.path.splitext(imageinfo['file_name'])[0] + '_' + str(i).zfill(2) + '.png'
            poly = translate(poly, x, y)
            poly = poly.buffer(0)
            if not poly.is_valid or poly.area == 0:
                print("无效图形" + imageinfo['file_name'] + '——' + str(i))
                continue
            frame = Polygon([[0, 0], [0, outsize], [outsize, outsize], [outsize, 0], [0, 0]])
            poly = poly.intersection(frame)
            # if poly.type == 'MultiPolygon' or poly.type == 'GeometryCollection':
            #     area = []
            #     for p in poly:
            #         area.append(p.area)
            #     poly = poly[area.index(max(area))]
            if poly.type == 'GeometryCollection':
                poly = MultiPolygon([Polygon(polygon) for polygon in poly if polygon.type == 'Polygon'])
            if not poly.is_valid or poly.area == 0:
                print("无效图形" + imageinfo['file_name'] + '——' + str(i))
                continue
            if poly.type == 'MultiPolygon':
                anns['segmentation'] = [np.transpose(np.array(polygon.exterior.coords.xy), [1, 0]).flatten().tolist() for polygon in poly]
            elif poly.type == 'Polygon':
                anns['segmentation'] = [np.transpose(np.array(poly.exterior.coords.xy), [1, 0]).flatten().tolist()]
            else:
                print("无效图形" + imageinfo['file_name'] + '——' + str(i))
                continue
            anns['id'] = id
            anns['image_id'] = id
            anns['bbox'] = [poly.bounds[0], poly.bounds[1], poly.bounds[2] - poly.bounds[0],
                            poly.bounds[3] - poly.bounds[1]]
            anns['area'] = poly.area
            images.append({'id': id, 'file_name': file_name, 'width': outsize, 'height': outsize,
                           'base_image': imageinfo['file_name'], 'base_image_id': imageinfo['id'], 'base_anns_id': i,
                           'translation': [x, y]})
            annotations.append(anns)
            cv2.imwrite(masksave + file_name, msk*255)
            plt.imsave(imagesave + file_name, img)
            print(file_name)
            id += 1
    with open(jsonsave + "cut.json", 'w') as jsonfile:
        jsonfile.write(
            json.dumps({'info': info, 'categories': categories, 'images': images, 'annotations': annotations}))


if __name__ == '__main__':
    # 处理数据
    mkr(imagesave)
    mkr(masksave)
    get_mask_data(650, 256)
    print('Got all the masks')
