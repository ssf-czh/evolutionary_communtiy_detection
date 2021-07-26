# -*- coding: utf-8 -*-
# @Time    : 2020/4/12 14:39
# @Author  : wmy
import csv
import os
import random

import networkx as nx

'''

利用二项分布根据图的拓扑结构及其社区结构生成节点属性。

生成方法：
        1.初始化属性个数a为社区个数k的m倍,并且初始值为0;
        2.遍历每个社区得到社区i的节点集合,遍历该节点集合,节点属性索引为[i:i+k)的每个属性值80%的概率为1,20%的概率为0,其余属性20%的概率为1,80%的概率为0;
        3.对于重叠节点,执行当前社区属性赋值时，对其他社区对应属性不做处理.
        
注: 拓扑结构文件格式: 一行一条边; 社区结构文件格式: 一行一个社区。

'''


class BDAttributeNetworkGenerator:
    def __init__(self, edge_path, community_path, multiple, zero_rate, one_rate):
        self.edge_path = edge_path  # 边路径
        self.community_path = community_path  # 真实社区路径
        self.attr_num = 0  # 属性个数,初始为0
        self.zero_rate = zero_rate  # 属性值为0的概率
        self.one_rate = one_rate  # 属性值为1的概率
        self.multiple = multiple  # 倍数: 节点属性个数/社区个数

    def read_graph(self):
        """
       根据边集读取图
       :return: 图
       """
        g = nx.read_edgelist(self.edge_path, nodetype=int)
        for node in g.nodes:
            attribute_list = [0] * self.attr_num  # 初始化节点属性向量值为0
            g.nodes[node]['feat'] = attribute_list
        return g

    def read_community(self):
        """
        读取图g的真实社区
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
        self.attr_num = community_num * self.multiple  # 设置节点属性个数 = 倍数*社区个数
        return community_list

    def genrate_attribute(self, g, community_list):
        """
        根据原始图拓扑结构以及社区信息为节点生成属性
        :param g: 原始图
        :param community_list: 社区列表
        :return g: 属性图（由于g属于传引用，g直接被修改,所以不需要返回g）
        """
        # 记录重叠节点及其所在社区索引
        intersect_nodes_dict = dict()
        for i in range(len(community_list)):
            for j in range(len(community_list)):
                if i < j:
                    intersect_nodes_list = set(community_list[i]).intersection(community_list[j])
                    for intersect_node in intersect_nodes_list:
                        if intersect_node not in intersect_nodes_dict.keys():
                            intersect_nodes_dict[intersect_node] = set()
                            intersect_nodes_dict[intersect_node].add(i)
                            intersect_nodes_dict[intersect_node].add(j)
                        else:
                            intersect_nodes_dict[intersect_node] = intersect_nodes_dict[intersect_node].union([i, j])

        for i in range(len(community_list)):
            for node in community_list[i]:
                attribute_list = g.nodes[node]['feat']
                comm_attr_idx = [j for j in
                                 range(i * self.multiple, i * self.multiple + self.multiple)]  # 当期那社区对应的属性索引
                remain_attr_idx = [i for i in range(self.attr_num) if i not in comm_attr_idx]  # 非当前社区对应的属性的索引
                if node not in intersect_nodes_dict.keys():  # 非重叠节点
                    for idx in comm_attr_idx:  # 为社区对应的属性赋值
                        attribute_list[idx] = self.random_index([self.zero_rate, self.one_rate])
                    for idx in remain_attr_idx:  # 为剩余的属性赋值
                        attribute_list[idx] = self.random_index([100 - self.zero_rate, 100 - self.one_rate])
                    g.nodes[node]['feat'] = attribute_list
                else:  # 重叠节点
                    for idx in comm_attr_idx:  # 为社区对应的属性赋值
                        attribute_list[idx] = self.random_index([self.zero_rate, self.one_rate])
                    dealed_idxs = list()  # 重叠节点其他社区对应属性的索引，不需要赋值
                    for comm_idx in intersect_nodes_dict[node]:  # 重叠节点所在社区索引
                        dealed_idx = [i for i in
                                      range(comm_idx * self.multiple,
                                            comm_idx * self.multiple + self.multiple)]
                        dealed_idxs += dealed_idx
                    for dealed_idx in dealed_idxs:  # 移除重叠节点其他社区对应属性的索引
                        if dealed_idx in remain_attr_idx:
                            remain_attr_idx.remove(dealed_idx)
                    for idx in remain_attr_idx:  # 为剩余的属性赋值，但是对节点其他社区对应属性不需要赋值
                        attribute_list[idx] = self.random_index([100 - self.zero_rate, 100 - self.one_rate])
                    g.nodes[node]['feat'] = attribute_list

    @staticmethod
    def random_index(rate):
        """
        :param rate: eg:[20, 80]  # 0的概率为20%，1的概率为80%
        :return: 概率事件的下标索引
        """
        start = 0
        index = 0
        randnum = random.randint(1, sum(rate))
        for index, scope in enumerate(rate):
            start += scope
            if randnum <= start:
                break
        return index

    def write_feat(self, g):
        """
        保存图的属性到本地文件(保存位置：拓扑文件同级目录，保存文件名：拓扑文件名_feat.txt)
        :param g: 属性图
        """
        # profix = os.path.dirname(self.edge_path) + '/' + str(os.path.basename(self.edge_path)[:-4])
        # file_name = profix + '_bd_feat' + '.txt'
        file_name = os.path.join(os.path.dirname(self.edge_path), 'temp.feat')
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
    multiple = 1 # 倍数
    zero_rate = 0  # 属性值为0的概率
    one_rate = 100 - zero_rate  # 属性值为1的概率
    # file_name_list = ['1k', '2k', '3k', '4k', '5k']
    # file_name_list = ['1k_0.7', '1k_0.8']
    # file_name_list = ['1k_0.1']
    # for file_name in file_name_list:
    #     # edge_path = '../datasets/attr/artificial/network' + file_name + '.txt'
    #     # community_path = '../datasets/attr/artificial/community' + file_name + '.txt'
    #     edge_path = os.path.join(file_name, 'citeseer.ugraph')
    #     community_path = os.path.join(file_name, 'citeseer.cmty')
    #     bdang = BDAttributeNetworkGenerator(edge_path, community_path, multiple, zero_rate, one_rate)
    #     bdang.run()

    edge_path = '../test/1k0.7.ugraph'
    community_path = '../test/1k0.7.cmty'

    bdang = BDAttributeNetworkGenerator(edge_path, community_path, multiple, zero_rate, one_rate)
    bdang.run()