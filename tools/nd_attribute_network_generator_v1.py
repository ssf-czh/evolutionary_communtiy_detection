# -*- coding: utf-8 -*-
# @Time    : 2020/4/12 14:39
# @Author  : wmy
import csv
import os
import random

import networkx as nx
import numpy as np

'''

利用正态分布根据图的拓扑结构及其社区结构生成节点属性。

生成方法：
        1.设置节点拥有的属性个数N;
        2.根据社区个数K生成K维正态分布的连续值，每一维的均值都不相同，方差相同;
        3.针对每一个社区中的每一个节点,根据当前社区索引index从第index维正态分布数据中采样N个连续值，作为当前社区节点的属性
        
注: 拓扑结构文件格式: 一行一条边; 社区结构文件格式: 一行一个社区。

'''


class NDAttributeNetworkGenerator:
    def __init__(self, edge_path, community_path, attr_num):
        self.edge_path = edge_path  # 边路径
        self.community_path = community_path  # 真实社区路径
        self.attr_num = attr_num  # 节点属性个数

    def read_community(self):
        """
        读取社区结构
        :return:社区集合
        """
        data = csv.reader(open(self.community_path), delimiter=' ')
        community_num = 0  # 记录社区个数
        community_list = []  # 存储社区
        for row in data:
            row = list(row)
            row = [int(x) for x in row]
            community_list.append(list(row))
            community_num += 1
        return community_list

    def read_graph(self):
        """
       根据边集读取图
       :return: 图
       """
        g = nx.read_edgelist(self.edge_path, nodetype=int)
        return g

    def genrate_attribute(self, g, community_list):
        """
        根据图拓扑结构以及真实社区生成节点属性
        :param g: 原始图
        :param community_list: 社区列表
        :return g: 属性图（由于g属于传引用，g直接被修改,所以不需要返回g）
        """
        k = len(community_list)
        mean = [i * 4 for i in range(k)]  # 均值数组
        v = [0.1 for i in range(k)]  # 协方差矩阵对角向量
        conv = np.diag(v)  # 根据向量构造协方差矩阵
        X = np.random.multivariate_normal(mean=mean, cov=conv, size=self.attr_num)  # 生成k维正态分布数据
        for i in range(len(community_list)):
            samples = X[:, i].tolist()  # 第i维正态分布样本集
            for node in community_list[i]:
                attribute_list = [0] * self.attr_num  # 初始化节点属性向量值为0
                for j in range(self.attr_num):
                    attribute_list[j] = random.sample(samples, 1)[0]  # 属性值采样
                g.nodes[node]['feat'] = attribute_list

    def write_feat(self, g):
        """
        保存图的属性到本地文件(保存位置：拓扑文件同级目录，保存文件名：拓扑文件名_feat.txt)
        :param g: 属性图
        """
        # profix = os.path.dirname(self.edge_path) + '/' + str(os.path.basename(self.edge_path)[:-4])
        # file_name = profix + '_bd_feat' + '.txt'
        file_name = os.path.join(os.path.dirname(self.edge_path), '1k0.7.nd2.feat')
        f = open(file_name, 'w')
        for node in sorted(g.nodes()):  # 顺序输出节点及其属性
            feat = g.nodes[node]['feat']
            obj = str(node)
            for item in feat:
                obj += ' ' + str(item)
            f.write(obj + '\n')
        f.close()
        print(file_name)

    def run(self):
        """
        程序执行入口
        """
        community_list = self.read_community()
        g = self.read_graph()
        self.genrate_attribute(g, community_list)
        self.write_feat(g)


if __name__ == '__main__':
    # attr_num = 16  # 节点属性个数
    # file_name_list = ['1k_0.1', '1k_0.2', '1k_0.3', '1k_0.4',
    #                   '1k_0.5', '1k_0.6', '1k_0.7']
    # for file_name in file_name_list:
    #     # edge_path = '../datasets/attr/artificial/network' + file_name + '.txt'
    #     # community_path = '../datasets/attr/artificial/community' + file_name + '.txt'
    #     edge_path = os.path.join(file_name, 'blog.ugraph')
    #     community_path = os.path.join(file_name, 'blog.cmty')
    #     ndang = NDAttributeNetworkGenerator(edge_path, community_path, attr_num)
    #     ndang.run()

    attr_num = 16
    edge_path = '../test/1k0.7.ugraph'
    community_path = '../test/1k0.7.cmty'

    ndang = NDAttributeNetworkGenerator(edge_path, community_path, attr_num)
    ndang.run()