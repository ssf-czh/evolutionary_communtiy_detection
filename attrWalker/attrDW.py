#! /usr/bin/env python
# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from collections import defaultdict
from six import iterkeys
import random
import gensim
from gensim import corpora
from tools import clustering
from tools import file_io
from metric import onmi,nmi, modularity
import metric.fitness as sim
from attrWalker.alias import Alias
import threading

def random_walk(alias,cur,length):
    walk = [' '] * (length + 1)
    walk[0] = str(cur)
    for i in range(1,length + 1):
        cur = alias.alias_sample(int(cur))
        walk[i] = str(cur)
    return walk

def capture_neighborhood_info(alias,cur,length = 5):
    center = length // 2
    walk = [' '] * length

    for i in range(center):
        neighbor = alias.alias_sample(cur)
        walk[i] = str(neighbor)

    walk[center] = str(cur)

    for i in range(center+1, length):
        neighbor = alias.alias_sample(cur)
        walk[i] = str(neighbor)

    return walk

# def create_alis():
    # ...

def build_deepwalk_corpus(nodeID, S,S_id2node, num_paths, path_length):

    alias = Alias()

    K = 30



    # S = np.dot(S,S)
    # S = np.dot(S, S)

    knearest_node = [0] * S.shape[0]
    knearest_sim = [0] * S.shape[0]

    node_map = list(map(lambda x:S_id2node[x],range(S.shape[0]) ))

    for i in range(S.shape[0]):
        rep_node = S_id2node[i]
        dist_with_index = zip(S[i, :], node_map)
        res = sorted(dist_with_index, key=lambda x: x[0], reverse=True)
        knearest_sim[rep_node] = list(map(lambda x:x[0],res))

        knearest_node[rep_node] = list(map(lambda x: str(x[1]), res))

        # knearest_node[rep_node] = list(map(lambda x: x[1], res))
        alias.create_alias_table(rep_node,
                                 knearest_node[rep_node][:35],knearest_sim[rep_node][:35])

        # 尝试一下对S归一化，再多乘几次.相当于提取了 更高阶信息?

        # 想法2:
        # wiki 200 80(numwalks) 50(numpath)
        # blog
        # 200 80(numwalks) 50(numpath) 0.4
        # 200 160 80 0.4 0.27
        # 100 80 50 0.44 0.3
        # 50 80 50 0.57 0.45
        # 60 80 50 0.56
        # 70 80 50 0.51 0.37
        # 40 80 50 0.57 0.45
        # 40 160 100 0.586 0.506
        # 50 200 100 0.525 0.398
        # 40 200 100 0.5669 0.4828
        # print(alias.alias_sample(rep_node))
        # alias.create_alias_table(rep_node,knearest_node[rep_node][:],knearest_sim[rep_node][:])

        # exit()
        # print(only_node)
        # exit()

    walks = []
    # nodes = list(G.nodes())

    print('walking..')


    # #想法2:
    # for cnt in range(200):
    #     for n in nodeID:
    #         walk = [' ']*100*2
    #         for j in range(100):
    #             walk[j*2] = str(n)
    #             nei = alias.alias_sample(n)
    #             walk[j*2 + 1] = nei
    #         # print(walk)
    #         # exit()
    #         walks.append(walk)


    for cnt in range(200):     # 对每个节点重复num_paths次游走
        # rand.shuffle(nodes)         # 打乱每个节点的顺序
        # print(len(nodeID))
        for n in nodeID:
            # k = random.randint(2,K)

            # walk = knearest_node[n][:20]
            # random.shuffle(walk)
            # walks.append( walk  )

            # walks.append( knearest_node[n][:10] + [str(n)] )
            walks.append(random_walk( alias, n, 200))
            # walks.append(capture_neighborhood_info(alias,n,5))

        # for node in nodes:
            # walks.append(G.random_walk(path_length, rand=rand, start=node))
    # # print(walks)
    return walks


def train(feat_path,output_path,S,S_id2node):

    number_walks = 80
    walk_length = 40
    window_size = 5

    representation_size = 128
    workers = 16

    feat_data = np.genfromtxt(feat_path, dtype=np.dtype(np.float32))
    nodeID = feat_data[:, 0].astype(int)
    vectors = np.delete(feat_data, 0, axis=1)
    map_id2node = {i: j for i, j in enumerate(nodeID)}
    map_node2id= {j: i for i, j in enumerate(nodeID)}

    print("Number of nodes: {}".format(vectors.shape[0]))

    num_walks = vectors.shape[0] * number_walks
    print("Number of walks: {}".format(num_walks))

    data_size = num_walks * walk_length
    print("Data size (walks*length): {}".format(data_size))


    print("Random Walking...")
    walks = build_deepwalk_corpus(nodeID, S,S_id2node, num_paths=number_walks, path_length=walk_length)

    print("Training...")
    model = Word2Vec(walks, vector_size=representation_size, window=window_size,
                     min_count=0, sg=1, hs=0, negative=5, workers=workers,
                     alpha=0.005)

    model.wv.save_word2vec_format(output_path)

    print("Training finished...")

    return model.wv.vectors



def attrDW( feat_path, comm_path, output_path, only_clustering=False,times=1):

    sum_onmi = 0
    max_onmi = 0
    sum_nmi = 0
    max_nmi = 0

    # feat_data = np.loadtxt(feat_path, dtype=float, delimiter=' ', skiprows=0)

    feat_data = np.genfromtxt(feat_path, dtype=np.dtype(np.float32))

    nodeID = feat_data[:, 0].astype(int)
    F = np.delete(feat_data, 0, axis=1)  # 裁掉第一列的 nodeID
    S_id2node = {i: j for i, j in enumerate(nodeID)}
    # for i in range(10):
    #     print(S_id2node[i])
    S = sim.gen_SimMatrix(F,'cosine')

    # G = nx.read_edgelist(net_path,nodetype=int)
    if comm_path is not None:
        real_comms = file_io.read_communities(comm_path, True)
        print('community number:', len(real_comms))

    for i in range(times):

        if only_clustering is False:
            train(feat_path,output_path,S,S_id2node)

        data = np.loadtxt(output_path, dtype=float, delimiter=' ', skiprows=1)

        nodeID = data[:, 0].astype(int)
        vectors = np.delete(data, 0, axis=1)  # 裁掉第一列的 nodeID
        map_id2node = {i: j for i, j in enumerate(nodeID)}



        if comm_path is not None:
            comms = clustering.kmeans_from_vec(vectors, len(real_comms), map_id2node, True)
            # comms = clustering.ward_from_vec(vectors, len(real_comms), map_id2node, True)
            csum = 0
            for c in comms:
                csum += len(c)
            print(csum)
            vNMI, vONMI = onmi.onmi(cover=comms, coverRef=real_comms,
                                    variant=None)
            nnmi = nmi.calc(comms, real_comms)

            sum_onmi += vONMI
            sum_nmi += nnmi
            if vONMI > max_onmi:
                max_onmi = vONMI
            if nnmi > max_nmi:
                max_nmi = nnmi

    print('max NMI:', max_nmi, ' avg NMI:', sum_nmi / times)
    print('max ONMI:', max_onmi, ' avg ONMI:', sum_onmi / times)



import os
if __name__ == '__main__':

    ds_name = 'cora'
    # print(os.system('ls ../dataset/wiki/remap/'))
    net = '../dataset/{:s}/remap/{:s}.ugraph'.format(ds_name, ds_name)
    com = '../dataset/{:s}/remap/{:s}.cmty'.format(ds_name, ds_name)
    feat = '../dataset/{:s}/remap/{:s}.feat'.format(ds_name, ds_name)
    # save_emb = './emb/Adw.{:s}.emb'.format(ds_name)
    save_emb = './emb/Adw.{:s}{:s}.emb'.format(ds_name, "no_rep")
    print(111)
    attrDW(feat,com,save_emb,only_clustering=False)