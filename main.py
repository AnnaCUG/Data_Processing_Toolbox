# -*-coding: utf-8 -*-
# @Time : 2022/4/26 16:34
# @Author : hewitt Wong
import polis
import cocoResultFormat
import cocoEvaluate


def run(kw=None):
    # GT 文件路径
    if kw is None:
        kw = ['P', 'C']
    Gt_path = r'H:\whtowerCode\PoLiS\datas\instances_test2014.json'
    # DT 文件路径
    Dt_path = r'H:\whtowerCode\PoLiS\datas\results_test_building'
    # PoLiS 评价结果保存路径
    p_save = r"./datas/polis.json"
    # DT 数据格式化后文件保存路径 改数据用于计算coco 评估
    format_save = r"./datas/format.json"
    # 数据格式化模式 J 传入json，P 传入图片
    type = 'J'
    # PoLiS 计算
    if 'P' in kw:
        polis.run(Gt_path, Dt_path, p_save)
    if 'C' in kw:
        # DT 数据格式化
        cocoResultFormat.run(Gt_path, Dt_path, format_save, type)
        # COCO评估
        cocoEvaluate.run(Gt_path, format_save)


if __name__ == '__main__':
    run()
