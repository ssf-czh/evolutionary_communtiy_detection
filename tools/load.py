#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   load.py
@Contact :   
@License :   

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/8/1 18:58   linsher      1.0         None
'''

import numpy as np
import torch
import scipy.sparse as sp
import networkx as nx
from tools import file_io
from collections import defaultdict

# set() 会把顺序随机打乱...
def load_data_attr_1(topology_path, attr_path ):
    '''
    support cora and citeseer
    :param topology_path:
    :param attr_path:
    :return:
    '''
    idx_features_labels = np.genfromtxt(attr_path,dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)

    nodes = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
    labels = np.array(idx_features_labels[:, -1], dtype=np.dtype(str))

    # features = sp.eye(labels.shape[0])

    nodes_uniq = list(set(nodes))
    labels_uniq = list(set(labels))
    nodes_uniq.sort()
    labels_uniq.sort()

    map_label2id = {j: i for i, j in enumerate(labels_uniq)}
    map_id2label = {i: j for i, j in enumerate(labels_uniq)}

    map_node2id = {j: i for i, j in enumerate(nodes_uniq)}
    map_id2node = {i: j for i, j in enumerate(nodes_uniq)}

    comms = defaultdict(list)
    for i in range(len(nodes)):
        n = map_node2id[nodes[i]]
        la = map_label2id[labels[i]]
        if la not in comms:
            comms[la] = [n]
        else:
            comms[la].append(n)

    real_comms = []
    for k,v in comms.items():
        real_comms.append(set(v))


    self_map = {i: i for i, j in enumerate(nodes_uniq)}

    edges_unordered = np.genfromtxt(topology_path,dtype=np.dtype(str))
    edges = np.array(list(map(map_node2id.get, edges_unordered.flatten())), dtype=np.int32)\
        .reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(nodes), len(nodes)),
                        dtype=np.float32)

    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # 把邻接矩阵对角线元素全部置为1。
    adj_targ = adj.todense() + sp.eye(adj.shape[0])

    # print(adj.todense())

    G = nx.Graph()
    G.add_edges_from(edges)






