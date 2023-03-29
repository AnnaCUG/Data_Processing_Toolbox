# -*-coding: utf-8 -*-
# @Time : 2022/4/26 14:09
# @Author : hewitt Wong

import os.path
import numpy as np
import json
import warnings
from shapely import geometry as geo

warnings.filterwarnings("ignore")


class PoLiS:
    def __init__(self):
        self._Dt_dir_path = None
        self._Gt_file = None
        self.un_match_count = 0
        self.polis = []

    def load_file(self, file_path, model):
        """加载数据

        :param file_path:GT json文件路径 或 DT 文件所在文件夹路径
        :param model:string: GT / DT
        :return: None
        """
        if model == "DT":
            self._Dt_dir_path = file_path
        elif model == "GT":
            self._Gt_file = json.load(open(file_path, 'r', encoding='utf-8'))
        else:
            exit("Class:PoLiS Func:load_file: Model Error")

    @staticmethod
    def _vector_cos(start2v, v2end, start2end):
        """
        计算边上两端点位置的余弦值，并判断是否为锐角
        :param start2v: 向量起始点到单独点的向量
        :param v2end: 单独点到向量终点的向量
        :param start2end: 多边形边所在的向量
        :return: 是否都是锐角，起始点所在角的余弦值，终点所在角的余弦值
        """
        # 公式：cos = (a*b)/(||a||*||b||)
        cos_start = (np.sum(start2v * start2end)) / (
                np.sqrt(np.sum(start2v ** 2)) * np.sqrt(np.sum(start2end ** 2)))
        cos_end = (np.sum(v2end * start2end)) / (np.sqrt(np.sum(v2end ** 2)) * np.sqrt(np.sum(start2end ** 2)))
        if cos_start < 0 or cos_end < 0:
            return False, cos_start, cos_end
        return True, cos_start, cos_end

    @staticmethod
    def min_dist_pt_2_poly(v, P):
        """
        计算点到多边形的最短距离
        :param v: 点坐标, np.array
        :param P: 多边形的顶点集合，np.array
        :return: 点到多边形的最短距离
        """
        # 点到点的最短距离
        dist2vertex = np.min(np.sqrt((v[0] - P[:, 0]) ** 2 + (v[1] - P[:, 1]) ** 2))
        # 点到边的距离
        dist2edge = []
        for i in range(P.shape[0] - 1):
            v_tmp_end = P[i + 1, :]
            v_tmp_start = P[i, :]
            start2v = v - v_tmp_start
            v2end = v_tmp_end - v
            start2end = v_tmp_end - v_tmp_start
            if np.sum(start2v ** 2) + np.sum(v2end ** 2) == np.sum(start2end ** 2):
                return 0  # 点在线上
            else:
                sign, cos_start, cos_end = PoLiS._vector_cos(start2v, v2end, start2end)
                if sign:
                    # 都是锐角，需要取垂线长
                    c_2 = np.sum(start2v ** 2)  # c^2
                    h = (c_2 - (cos_start * (c_2 ** 0.5)) ** 2) ** 0.5
                    dist2edge.append(h)
        if dist2edge.__len__() != 0:
            return min(min(dist2edge), dist2vertex)
        return dist2vertex

    @staticmethod
    def PoLiS_metric(A, B):
        """
        计算PoLiS
        :param A: 多边形A
        :param B: 多边形B
        :return: PoLiS值
        """
        _p1 = 0.0
        _p2 = 0.0
        length_of_A = A.shape[0]
        length_of_B = B.shape[0]
        for j in range(length_of_A):
            _p1 = _p1 + PoLiS.min_dist_pt_2_poly(A[j], B)
        for k in range(length_of_B):
            _p2 = _p2 + PoLiS.min_dist_pt_2_poly(B[k], A)
        p = (_p1 / (2.0 * length_of_A)) + (_p2 / (2.0 * length_of_B))
        return p

    @staticmethod
    def polygon_IOU(polygon_Gt, polygon_Dt):
        Gt_polygon = []
        Dt_polygon = []
        for i in polygon_Gt:
            tmp_list = np.asarray(i).reshape((-1, 2))
            Gt_polygon.append(geo.Polygon(tmp_list).area)
        for i in polygon_Dt:
            tmp_list = np.asarray(i).reshape((-1, 2))
            Dt_polygon.append(geo.Polygon(tmp_list).area)
        Gt_m_polygon = geo.Polygon(np.asarray(polygon_Gt[np.argmax(Gt_polygon)]).reshape((-1, 2)))
        Dt_m_polygon = geo.Polygon(np.asarray(polygon_Dt[np.argmax(Dt_polygon)]).reshape((-1, 2)))
        inters = Gt_m_polygon.intersection(Dt_m_polygon).area  # 交集
        sym = Gt_m_polygon.union(Dt_m_polygon).area  # 并集
        return inters / sym

    @staticmethod
    def max_area(Dt):
        Dt_polygon_area = []
        for i in Dt:
            Dt_polygon_area.append(geo.Polygon(np.asarray(i).reshape((-1, 2))).area)
        return np.asarray(Dt[np.argmax(Dt_polygon_area)]).reshape((-1, 2))

    def calculate(self):
        for idx in range(self._Gt_file['annotations'].__len__()):
            Gt_ann = self._Gt_file['annotations'][idx]
            tmp_iou = []
            tmp_Dt_file_name = self._Gt_file['images'][int(Gt_ann['image_id']) - 1]['file_name']
            Dt_file_path = ''.join(os.path.join(self._Dt_dir_path, tmp_Dt_file_name).split('.')[:-1]) + '.json'
            Dt_file = json.load(open(Dt_file_path))
            Gt_polygon_list = Gt_ann['segmentation']
            for i in range(Dt_file['object'].__len__()):
                Dt_polygon_list = Dt_file['object'][i]['mask']
                t_iou = PoLiS.polygon_IOU(Gt_polygon_list, Dt_polygon_list)
                tmp_iou.append(t_iou)
            if tmp_iou.__len__() > 0:
                if max(tmp_iou) > 0.7:  # 匹配到了!
                    index_ = tmp_iou.index(max(tmp_iou))
                    t_Dt_polygon_mask = PoLiS.max_area(Dt_file['object'][index_]['mask'])
                    t_Gt_polygon_mask = PoLiS.max_area(Gt_ann['segmentation'])
                    s = PoLiS.PoLiS_metric(t_Gt_polygon_mask, t_Dt_polygon_mask)
                    print(f"id:{Gt_ann['id']}\tIOU:{max(tmp_iou):.5f}\tPoLiS:{s:.5f}")
                    self.polis.append(s)
                else:
                    print(f"id:{Gt_ann['id']}\tun matched")
                    self.un_match_count += 1
            else:
                print(f"id:{Gt_ann['id']}\tun matched")
                self.un_match_count += 1

    def get_result(self, save_path):
        mean = np.mean(np.asarray(self.polis))
        print('average PoLiS: ', mean)
        print('un matched count: ', self.un_match_count)
        print('all count: ', self._Gt_file['annotations'].__len__())
        with open(save_path, 'w', encoding='utf-8') as json_file:
            f = {'PoLiS': self.polis, 'average PoLiS': mean, 'unmatched count': self.un_match_count,
                 'all count': self._Gt_file['annotations'].__len__()}
            json.dump(f, json_file, ensure_ascii=False)
        return self.polis, self.un_match_count, self._Gt_file['annotations'].__len__()


def run(gt_path, dt_path, save_path):
    p = PoLiS()
    p.load_file(gt_path, 'GT')
    p.load_file(dt_path, 'DT')
    p.calculate()
    p.get_result(save_path)


if __name__ == '__main__':
    run(r'H:\whtowerCode\PoLiS\datas\instances_test2014.json', r'H:\whtowerCode\PoLiS\datas\results_test_building',
        r"./datas/polises.json")
