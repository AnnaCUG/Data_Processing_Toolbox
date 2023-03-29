import os
import json



info = {"about": "Dataset for SpaceNet Building Detection Challenge", "date_created": "06/14/2022", "description": "SpaceNet 2 AOI_2_Vegas dataset"}
categories = [{"id": 1, "name": "building", "supercategory": "building"}]
json_dir = "D:/boundary-optimized/Mask_RCNN_X_spacenet/json_results/"
files = os.listdir(json_dir)
id = 0
images = []
annotations = []
for image_id, file in enumerate(files):
    with open(json_dir + file, 'r') as jsonfile:
        json_file = json.load(jsonfile)
    images.append({'id': image_id, 'file_name': json_file['images']['file_name'], 'width': json_file['images']['width'],
                   'height': json_file['images']['width']})
    for annotation in json_file['annotations']:
        annotations.append({'id': id, 'image_id': image_id, 'category_id': 1, 'segmentation': annotation['segmentation'],
                       'bbox': annotation['bbox']})
        id += 1
with open("D:/boundary-optimized/Mask_RCNN_X_spacenet/out.json", 'w') as jsonfile:
    jsonfile.write(
        json.dumps({'info': info, 'categories': categories, 'images': images, 'annotations': annotations}))