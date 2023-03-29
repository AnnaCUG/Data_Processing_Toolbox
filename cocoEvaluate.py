# -*-coding: utf-8 -*-
# @Time : 2022/4/27 12:43
# @Author : hewitt Wong
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def run(Gt_path, Dt_path, type='segm'):
    cocoGt = COCO(Gt_path)
    cocoDt = cocoGt.loadRes(Dt_path)
    cocoEval = COCOeval(cocoGt, cocoDt, type)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    Gt_path = r'H:\whtowerCode\PoLiS\datas\instances_test2014.json'
    Dt_path = r"./datas/format.json"
    run(Gt_path, Dt_path)
