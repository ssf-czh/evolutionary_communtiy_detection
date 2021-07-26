# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea

import time
import networkx as nx
import numpy as np
import math
from metric import modularity,nmi,onmi
from scipy.spatial.distance import cdist
from tools import file_io
import metric.fitness as sim
import matplotlib.pyplot as plt
from tools.Loader import Loader
from tools import utils
from tools.clustering import kmeans_from_vec,ward_from_vec,kmeans_
from sklearn.cluster import SpectralClustering
from sklearn import preprocessing
from collections import defaultdict
# from comparison import deepwalk_qinze as dw
from tools import clustering



class myMOProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, ds_name, useSSE = True, M=2):
        # print(ds_name)
        # ds_name = 'cornell'
        net = '../dataset/{:s}/remap/{:s}.ugraph'.format(ds_name, ds_name)
        com = '../dataset/{:s}/remap/{:s}.cmty'.format(ds_name, ds_name)
        feat = '../dataset/{:s}/remap/{:s}.feat'.format(ds_name, ds_name)

        loader = Loader(net, feat)
        G, F0 = loader.get_GFMC()

        # emb_path = '../KNNGraph/emb/dw.{:s}.K{:d}.emb'.format

        attr_emb_path = '../attrWalker/emb/Adw.{:s}.emb'.format(ds_name)
        topo_emb_path = '../emb/dw.{:s}.emb'.format(ds_name)

        self.useSSE = useSSE
        if useSSE:
            data = np.loadtxt(attr_emb_path, dtype=float, delimiter=' ', skiprows=1)
            AnodeID = data[:, 0].astype(int)
            Avectors = np.delete(data, 0, axis=1)  # 裁掉第一列的 nodeID
            self.Amap_id2node = {i: j for i, j in enumerate(AnodeID)}
            self.Amap_node2id = {j: i for i, j in enumerate(AnodeID)}

            data = np.loadtxt(topo_emb_path, dtype=float, delimiter=' ', skiprows=1)
            TnodeID = data[:, 0].astype(int)
            Tvectors = np.delete(data, 0, axis=1)  # 裁掉第一列的 nodeID
            self.Tmap_id2node = {i: j for i, j in enumerate(TnodeID)}
            self.Tmap_node2id = {j: i for i, j in enumerate(TnodeID)}

        else:
            Avectors = F0
            self.Amap_id2node, self.Amap_node2id = loader.get_map()

        self.AF = Avectors
        self.TF = Tvectors

        # 节点属性.
        self.F = Avectors
        self.S = sim.gen_SimMatrix(self.F, 'cosine')

        self.real_comm = file_io.read_communities(com, True)
        k = self.k = len(self.real_comm)

        # 节点属性:归一化嵌入向量
        self.dists = np.zeros((self.F.shape[0], k))
        self.normF = np.linalg.norm(self.F,axis=1)

        min_node = 0
        max_node = G.number_of_nodes() - 1

        name = 'MOfindCenter'  # 初始化name（函数名称，可以随意设置）
        M = M  # 初始化M（目标维数）

        # 用原始数据就最大化cos，降维后就最小化sse
        if not useSSE:
            # 用模块度和COS. 最大化
            maxormins = [-1, -1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        else:
            # 用两个嵌入的SSE. 最小化
            maxormins = [1, 1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）

        Dim = k  # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [min_node] * Dim  # 决策变量下界
        ub = [max_node] * Dim # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        x = pop.Phen  # 得到决策变量矩阵

        objV = np.zeros((x.shape[0],2))

        # 如果不适用SSE 使用模块度和原属性cos
        if not self.useSSE:
            for z in range(x.shape[0]):
                xz = x[z,:]
                center_nodes = list(xz)
                comm_key = defaultdict(list)

                for i in range(self.S.shape[0]):
                    max_sim = 0
                    max_cid = 0
                    for j in center_nodes:
                        if i == j:
                            max_cid = j
                            break
                        elif self.S[i,j] > max_sim:
                            max_sim = self.S[i,j]
                            max_cid = j
                    # s_list = self.S[i, center_nodes]
                    # cid = np.argmax(s_list)
                    if max_cid not in comm_key:
                        comm_key[max_cid] = [i]
                    else:
                        comm_key[max_cid].append(i)

                comms = []
                for k, v in comm_key.items():
                    comms.append(v)
                # if len(comms) != x.shape[1]:
                #     print('?')
                #     print(self.S[center_nodes,center_nodes])
                #     print(comm_key.keys())
                #     print(center_nodes)
                #     print(len(comms))
                #     print(x.shape[1])
                #     exit()

                cos = sim.cal_simCos(comms, self.F)
                mod = modularity.cal_EQ(comms,self.real_comm)
                objV[z] = [mod,cos]

        # 如果使用降维后的属性 计算 intertia. 使用欧氏距离 但不能是0-1型嵌入

        if self.useSSE:
            for z in range(x.shape[0]):
                xz = x[z,:]
                # 取该个体的中心节点
                # xz = map(xz,self.map_node2id)
                center_vec_A = self.AF[xz]
                center_vec_T = self.TF[xz]

                # 计算所有个体到每个中心节点的距离
                # attr
                col = 0
                for c in center_vec_A:
                    distances = np.sum(np.asarray(self.AF - c) ** 2, axis=1)
                    self.dists[:, col] = distances
                    col += 1
                # 取个体到最近中心的距离. 并求和
                mindst = np.min(self.dists, axis=1)
                score_A = np.sum(mindst)

                # topo
                col = 0
                for c in center_vec_T:
                    distances = np.sum(np.asarray(self.TF - c) ** 2, axis=1)
                    self.dists[:, col] = distances
                    col += 1
                # 取个体到最近中心的距离. 并求和
                mindst = np.min(self.dists, axis=1)
                score_T = np.sum(mindst)

                objV[z] = [score_A, score_T]

        # 计算 intertia. 使用余弦相似度 0-1嵌入也可以了，但是还是比较差.
        # for z in range(x.shape[0]):
        #     xz = x[z,:]
        #     center_vec = self.F[xz]
        #
        #     col = 0
        #
        #     for c in center_vec:
        #         cnorm = np.linalg.norm(c)
        #         dot_res = np.inner(self.F , c)
        #         dot_res = dot_res / (self.normF * cnorm)
        #         self.dists[:, col] = 1 - dot_res
        #
        #         col += 1
        #     mindst = np.min(self.dists, axis=1)
        #     score = np.sum(mindst)
        #
        #     objV[z] = score


        pop.ObjV = objV
