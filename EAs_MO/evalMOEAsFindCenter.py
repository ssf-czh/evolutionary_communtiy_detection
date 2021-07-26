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
from sklearn.cluster import SpectralClustering, KMeans
from sklearn import preprocessing
from collections import defaultdict
# from comparison import deepwalk_qinze as dw
from tools import clustering

def MOEA_FindCenter_eval_Q_NMI(emb_attr_path, topo_attr_path, Phen_path,
                                         net_path,feat_path, real_comm_path, useSSE = True):

    data_attr = np.loadtxt(emb_attr_path, dtype=float, delimiter=' ', skiprows=1)
    nodeID_attr = data_attr[:, 0].astype(int)
    vectors_attr = np.delete(data_attr, 0, axis=1)
    map_id2node_attr = {i: j for i, j in enumerate(nodeID_attr)}
    map_node2id_attr = {j: i for i, j in enumerate(nodeID_attr)}

    data_topo = np.loadtxt(topo_attr_path, dtype=float, delimiter=' ', skiprows=1)
    nodeID_topo = data_topo[:, 0].astype(int)
    vectors_topo = np.delete(data_topo, 0, axis=1)
    map_id2node_topo = {i: j for i, j in enumerate(nodeID_topo)}
    map_node2id_topo = {j: i for i, j in enumerate(nodeID_topo)}


    phen = np.loadtxt(Phen_path, dtype=float, delimiter=',', skiprows=0).astype(int)
    print(phen.shape)

    # 重映射到节点
    # phen_node = np.zeros_like(phen)
    # for i in range(phen.shape[0]):
    #     phen_node[i,:] = map(phen[i,:],map_id2node_attr)

    # G = nx.read_edgelist(net_path,nodetype=int)
    loader = Loader(net_path, feat_path)
    G, F0 = loader.get_GFMC()

    real_comms = file_io.read_communities(real_comm_path,True)
    K = len(real_comms)

    maxNMI_T = 0
    maxNMI_A = 0

    # print(phen.shape)
    # exit()
    for z in range(len(phen)):
        center_nodes = list(phen[z,:])

        center_vec_A = vectors_attr[center_nodes]
        center_vec_T = vectors_topo[center_nodes]

        # 用emb做嵌入的话
        # print(center_vec.shape)
        # print(center_nodes)

        # 如果使用降维后的属性. 根据中心点. 直接使用kmeans生成社区

        comms_A, clf_A = kmeans_(vectors_attr, K, map_id2node= map_id2node_attr, n_init=1, init=center_vec_A)
        comms_T, clf_T = kmeans_(vectors_topo, K, map_id2node= map_id2node_topo, n_init=1, init=center_vec_T)



        # 如果不使用SSE . 根据cos计算最近的中心点. 生成社区
        if not useSSE:
            S = sim.gen_SimMatrix(F0, 'cosine')
            comm_key = {}
            for i in range(S.shape[0]):
                max_sim = 0
                max_cid = 0
                for j in center_nodes:
                    if i == j:
                        max_sim = 1
                        max_cid = j
                    elif S[i,j] > max_sim:
                        max_sim = S[i,j]
                        max_cid = j

                if max_cid not in comm_key:
                    comm_key[max_cid] = [i]
                else:
                    comm_key[max_cid].append(i)

            comms = []
            for k, v in comm_key.items():
                comms.append(v)


        vNMI = nmi.calc(comms_A,real_comms)
        if vNMI > maxNMI_A:
            maxNMI_A = vNMI

        vNMI = nmi.calc(comms_T,real_comms)
        if vNMI > maxNMI_T:
            maxNMI_T = vNMI


    print('max NMI using topo:', maxNMI_T)
    print('max NMI using attr:', maxNMI_A)

    #计算interia
    # col = 0
    # dists = np.zeros((vectors_attr.shape[0], len(center_nodes)))
    # # 计算所有个体到每个中心节点的距离
    # for c in center_vec:
    #     distances = np.sum(np.asarray(vectors_attr - c) ** 2, axis=1)
    #     # distances = np.sqrt(np.sum(np.asarray(vec_data - c) ** 2, axis=1))
    #     dists[:, col] = distances
    #     col += 1
    # # 取个体到最近中心的距离. 并求和
    # mindst = np.min(dists, axis=1)
    # inertia = np.sum(mindst)
    # print(inertia)


    # 根据cos计算最近的中心点. 计算inertia
    # col = 0
    # dists = np.zeros((vectors_attr.shape[0], len(center_nodes)))
    # normF = np.linalg.norm(vectors_attr, axis=1)
    # for c in center_vec:
    #     cnorm = np.linalg.norm(c)
    #     dot_res = np.inner(vectors_attr , c)
    #     dot_res = dot_res / (normF * cnorm)
    #     dists[:, col] = 1 - dot_res
    #
    #     col += 1
    # mindst = np.min(dists, axis=1)
    # inertia = np.sum(mindst)
    # print(inertia)





def evalMOEAsFindCenter(ds_name, useSSE):
    """
    :param ds_name: 数据集名
    :param useSSE:  是否使用嵌入属性算SSE
    :return:
    """
    net = '../dataset/{:s}/remap/{:s}.ugraph'.format(ds_name, ds_name)
    com = '../dataset/{:s}/remap/{:s}.cmty'.format(ds_name, ds_name)
    feat = '../dataset/{:s}/remap/{:s}.feat'.format(ds_name, ds_name)
    emb_attr_path = '../attrWalker/emb/Adw.{:s}.emb'.format(ds_name)
    emb_topo_path = '../emb/dw.{:s}.emb'.format(ds_name)
    Phen_path = './Result/Phen.csv'
    MOEA_FindCenter_eval_Q_NMI(emb_attr_path,emb_topo_path,
                                         Phen_path,net,feat,com,useSSE=useSSE )

if __name__ == '__main__':
    ds_name = 'cora'

    net = '../dataset/{:s}/remap/{:s}.ugraph'.format(ds_name, ds_name)
    com = '../dataset/{:s}/remap/{:s}.cmty'.format(ds_name, ds_name)
    feat = '../dataset/{:s}/remap/{:s}.feat'.format(ds_name, ds_name)
    emb_attr_path = '../attrWalker/emb/Adw.{:s}.emb'.format(ds_name)
    # emb_attr_path = '../attrWalker/emb/Adw.{:s}{:s}.emb'.format(ds_name,"no_rep")
    emb_topo_path = '../emb/dw.{:s}.emb'.format(ds_name)
    Phen_path = './Result/Phen.csv'

    MOEA_FindCenter_eval_Q_NMI(emb_attr_path,emb_topo_path,
                                         Phen_path,net,feat,com, useSSE = True)
