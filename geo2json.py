from osgeo import gdal
from osgeo import osr
import numpy as np
import json
from shapely.geometry import Polygon, MultiPolygon
import os
# 对应自己的python包的安装地址
os.environ['PROJ_LIB'] = r'C:/Users/HAN/anaconda3/Library/share/proj'

def geo2imagexy(dataset, x, y):
    '''
    根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
    :param dataset: GDAL地理数据
    :param x: 投影或地理坐标x
    :param y: 投影或地理坐标y
    :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    '''
    trans = dataset.GetGeoTransform()
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解

if __name__ == '__main__':
    images = []
    annotations = []
    image_id = 0
    id = 0
    gdal.AllRegister()
    img_dir = "G:/RS-building/RS-building/AOI_2_Vegas_Train/RGB-PanSharpen/"
    geo_dir = "G:/RS-building/RS-building/AOI_2_Vegas_Train/geojson/buildings/"
    img_files = os.listdir(img_dir)
    geo_files = os.listdir(geo_dir)
    for img_file, geo_file in zip(img_files, geo_files):
        dataset = gdal.Open(img_dir + img_file)
        with open(geo_dir + geo_file, 'r') as load_json:
            json_dict = json.load(load_json)
        buliding_number = 0
        for feature in json_dict['features']:
            segmentation = []
            type = feature['geometry']['type']
            coordinates = feature['geometry']['coordinates']
            if type == "Polygon":
                poly = []
                for point in coordinates[0]:
                    coords = geo2imagexy(dataset, point[0], point[1])
                    poly.extend(coords)
                segmentation.append(poly)
                p = Polygon(np.array(poly).reshape([-1, 2]))
            elif type == "MultiPolygon":
                for polygon in coordinates:
                    poly = []
                    for point in polygon[0]:
                        coords = geo2imagexy(dataset, point[0], point[1])
                        poly.extend(coords)
                    segmentation.append(poly)
                c = [[np.array(poly).reshape([-1, 2]).tolist()] for poly in segmentation]
                p = MultiPolygon([Polygon(np.array(poly).reshape([-1, 2])) for poly in segmentation])
            else:
                continue
            bbox = [p.bounds[0], p.bounds[1], p.bounds[2] - p.bounds[0], p.bounds[3] - p.bounds[1]]
            annotations.append({'area': p.area, 'iscrowd': 0, 'id': id, 'image_id': image_id, 'category_id': 1, 'segmentation': segmentation, 'bbox': bbox})
            id += 1
            buliding_number += 1
        if buliding_number:
            images.append(
                {'id': image_id, 'file_name': os.path.splitext(img_file)[0] + ".jpg", 'width': dataset.RasterXSize,
                 'height': dataset.RasterYSize})
            image_id += 1
        else:
            print(img_file)
    with open("D:/spacenet/annotations/instances_train2014.json", 'w') as jsonfile:
        jsonfile.write(json.dumps({'images': images, 'annotations': annotations}))
