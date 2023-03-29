import json
import os.path
import cv2
import numpy as np
from osgeo import gdal
import shapefile
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.affinity import scale, translate

inputs_tif = [r"D:/HAN/WUH/big_tif/test.tif", r"D:/HAN/WUH/big_tif/test3.tif"] #打开原始tiff文件的路径
output_tif = r"D:/HAN/WUH/cut/test/tif/" #保存裁剪后的tiff文件的路径
output_jpg = r"D:/HAN/WUH/cut/test/jpg/"
input_shp = r"D:/HAN/WUH/2. The shape file of the whole area/allbuilding.shp"
output_json = r"D:/HAN/WUH/cut/test.json"
clip_width = 2048 #裁剪宽度
clip_height = 2048 #裁剪高度
step_width = 1024 #滑动宽度
step_height = 1024 #滑动高度
out_width = 512 #输出宽度
out_height = 512 #输出高度

if not os.path.exists(output_tif):
    os.makedirs(output_tif)
if not os.path.exists(output_jpg):
    os.makedirs(output_jpg)
# 为了支持中文路径，请添加下面这句代码
gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
# 为了使属性表字段支持中文，请添加下面这句
gdal.SetConfigOption("SHAPE_ENCODING", "")


def clip():
    images = []
    annotations = []
    image_id = 0
    annotation_id = 0
    shape_file = shapefile.Reader(input_shp)
    shapes = []
    for shape in shape_file.shapeRecords():
        shape = shape.shape.__geo_interface__
        if len(shape['coordinates']) < 1:
            continue
        if len(shape['coordinates'][0]) < 3:
            continue
        if len(shape['coordinates']) > 1:
            polygon = Polygon(shape['coordinates'][0],shape['coordinates'][1:])
        else:
            polygon = Polygon(shape['coordinates'][0])
        if not polygon.is_valid:
            print("异常图形")
            continue
        shapes.append(polygon)
    shapes = MultiPolygon(shapes)
    for input_tif in inputs_tif:
        file_name = os.path.splitext(os.path.split(input_tif)[-1])[0]
        dataset = gdal.Open(input_tif)
        input_width = dataset.RasterXSize
        input_height = dataset.RasterYSize
        input_bands = dataset.RasterCount
        geoTransform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()
        for start_w in range(0, input_width - clip_width + step_width, step_width):
            for start_h in range(0, input_height - clip_height + step_height, step_height):
                if start_w + clip_width > input_width:
                    start_w = input_width - clip_width
                if start_h + clip_height > input_height:
                    start_h = input_height - clip_height
                start_x = geoTransform[0] + start_w * geoTransform[1]
                start_y = geoTransform[3] + start_h * geoTransform[5]
                end_x = start_x + clip_width * geoTransform[1]
                end_y = start_y + clip_height * geoTransform[5]
                pixel_w = geoTransform[1] * clip_width / out_width
                pixel_h = geoTransform[5] * clip_height / out_height
                output_geoTransform = (start_x, pixel_w, geoTransform[2], start_y, geoTransform[4], pixel_h)
                tif_clip = []
                for band in range(input_bands):
                    band_data = dataset.GetRasterBand(band + 1)
                    tif_clip.append(band_data.ReadAsArray(start_w, start_h, clip_width, clip_height))
                tif_clip = np.asarray(tif_clip)
                tif_clip = cv2.resize(tif_clip.transpose((2, 1, 0)).astype(float), (out_width, out_height)).transpose((2, 1, 0))
                tif_clip_name = file_name + "_clip_" + str(start_w).zfill(6) + "_" + str(start_h).zfill(6) + ".tif"
                tif_outpath = os.path.join(output_tif, tif_clip_name)
                tif_out = gdal.GetDriverByName("GTiff").Create(tif_outpath, out_width, out_height, input_bands, gdal.GDT_Float32)
                tif_out.SetGeoTransform(output_geoTransform)
                tif_out.SetProjection(projection)
                for k in range(input_bands):
                    tif_out.GetRasterBand(k + 1).WriteArray(tif_clip[k, :, :])
                tif_out.FlushCache()
                for k in range(input_bands):
                    tif_out.GetRasterBand(k + 1).ComputeStatistics(False)
                tif_out.BuildOverviews('average', [2, 4, 8, 16, 32])
                print(tif_clip_name)
                del tif_out
                jpg_clip_name = file_name + "_clip_" + str(start_w).zfill(6) + "_" + str(start_h).zfill(6) + ".jpg"
                jpg_outpath = os.path.join(output_jpg, jpg_clip_name)
                cv2.imwrite(jpg_outpath, tif_clip[[2,1,0]].transpose((1, 2, 0)))
                frame = Polygon([[start_x, start_y], [end_x, start_y], [end_x, end_y], [start_x, end_y], [start_x, start_y]])
                frame = scale(frame, 1.01, 1.01)
                polygon_cilp_list = []
                for polygon in shapes:
                    polygon_clip = polygon.intersection(frame)
                    if polygon_clip.type=='MultiPolygon':
                        for pcp in polygon_clip:
                            polygon_cilp_list.append(pcp)
                    elif polygon_clip.type=='Polygon':
                        polygon_cilp_list.append(polygon_clip)
                shapes_cilp = MultiPolygon(polygon_cilp_list)
                if shapes_cilp.area == 0:
                    continue
                shapes_cilp = translate(shapes_cilp, -start_x, -start_y)
                shapes_cilp = scale(shapes_cilp, 1 / pixel_w, 1 / pixel_h, origin=(0, 0))
                frame = Polygon([[0, 0], [out_width, 0], [out_width, out_height], [0, out_height], [0, 0]])
                polygon_cilp_list = []
                for polygon in shapes_cilp:
                    polygon_clip = polygon.intersection(frame)
                    if polygon_clip.type == 'MultiPolygon':
                        for pcp in polygon_clip:
                            polygon_cilp_list.append(pcp)
                    elif polygon_clip.type == 'Polygon':
                        polygon_cilp_list.append(polygon_clip)
                shapes_cilp = MultiPolygon(polygon_cilp_list)
                if shapes_cilp.area == 0:
                    continue
                for polygon in shapes_cilp:
                    segmentation = []
                    segmentation.append(np.transpose(np.array(polygon.exterior.coords.xy), [1, 0]).flatten().tolist())
                    for interior in polygon.interiors:
                        segmentation.append(np.transpose(np.array(interior.coords.xy), [1, 0]).flatten().tolist())
                    bbox = [polygon.bounds[0], polygon.bounds[1], polygon.bounds[2] - polygon.bounds[0], polygon.bounds[3] - polygon.bounds[1]]
                    annotations.append({'id': annotation_id, 'image_id': image_id, 'category_id': 1, 'area': polygon.area, 'bbox': bbox, 'segmentation': segmentation})
                    annotation_id += 1
                images.append({'id': image_id, 'file_name': jpg_clip_name, 'width': out_width, 'height': out_height, 'clip_x': start_w, 'clip_y': start_h, 'clip_w': clip_width, 'clip_h': clip_height, 'start_x': start_x, 'start_y': start_y, 'pixel_w': pixel_w, 'pixel_h': pixel_h})
                image_id += 1
    info = {'description': 'WUH Dataset Clip', 'version': 1.0, 'year': 2014,'date_created': '2022/11/04'}
    categories = [{'id': 1, 'name': 'building', 'supercategory': 'building'}]
    with open(output_json, 'w') as jsonfile:
        jsonfile.write(
            json.dumps({'info': info, 'categories': categories, 'images': images, 'annotations': annotations}))

if __name__ == '__main__':
    clip()
