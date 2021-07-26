import networkx as nx
import numpy as np
import math
import matplotlib.pyplot as plt

class Loader :
    def __init__(self, net_file, feat_file, given_map_node2id=None):
        self.G = nx.read_edgelist(net_file,nodetype=int)

        raw_node_features = np.genfromtxt(feat_file, dtype=np.dtype(np.float32))

        # this way is much faster than self.feat = raw_node_feat[:,1:].. why?
        # self.feat still points to raw_node_feat? which makes it slower?
        # linsher 20201203
        #self.features = np.array(raw_node_features[:, 1:], dtype=np.int32) # slow method

        self.features = np.zeros((raw_node_features.shape[0], raw_node_features.shape[1] - 1))

        # given_map_node2id 期望节点n在生成矩阵的第几行
        # 没有给定 given_map_node2id 按文件顺序排序
        if given_map_node2id is None:
            for i in range(raw_node_features.shape[1] - 1):
                self.features[:, i] = raw_node_features[:, i + 1]

            self.map_id2node = raw_node_features[:, 0].astype(int)
            self.map_node2id = {n: i for i, n in enumerate(self.map_id2node)}
        else:
        # 根据 given_map_node2id 对索引重新排序
            for i in range(raw_node_features.shape[0]):
                rep_node = raw_node_features[i,0]
                self.features[given_map_node2id[rep_node],:] = raw_node_features[i, 1:]

            # in this case, we should rebuild map_n2id and map_id2n
            self.map_node2id = given_map_node2id
            self.map_id2node = {}
            for k,v in given_map_node2id.items():
                self.map_id2node[v] = k

    def get_map(self):
        return self.map_id2node, self.map_node2id

    def get_features_idmap(self):
        # id2node
        return self.map_id2node
    def get_features_n2id(self):
        # node2id
        return self.map_node2id

    def get_GFMC(self):
        return self.G, self.features

    def filter_feat(self, threshold = 0.1):
        if threshold == 0.0:
            return self.features

        F = (self.features.sum(0) / self.features.shape[0]).tolist()
        elist = []
        cntr = 0
        for i in range(len(F)):
            v = F[i]
            w = 1 - v
            e = 0
            if v > 0: e += - v * math.log(v, 2)
            if w > 0: e += - w * math.log(w, 2)
            if e >= threshold:
                elist.append(i)
                cntr += 1

        # this way is much faster than newF = self.feat[:,elist]
        newF = np.zeros((self.features.shape[0], cntr),dtype=np.float32)
        fid = 0
        for each in elist:
            newF[:,fid] = self.features[:,each]
            fid += 1
        # newF = self.features[:,elist]

        return newF


    def show_feat_entropy_hist(self):
        F = self.features.sum(0).tolist()
        elist = []
        for i in range(len(F)):
            v = F[i] / F.shape[0]
            w = 1 - v
            e = 0
            if v > 0: e += - v * math.log(v, 2)
            if w > 0: e += - w * math.log(w, 2)
            elist.append(e)
        plt.hist(elist,  50)
        plt.show()

