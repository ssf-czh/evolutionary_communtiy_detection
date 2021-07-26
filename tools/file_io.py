#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   file_io.py    
@Contact :   
@License :   

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/6/14 15:09   linsher      1.0         None
'''

def save_edges(edgeList, path):
    with open(path,'w') as f:
        for e in edgeList:
            f.write(e[0] + '\t' + e[1] + '\n')
    print('edge list is written to {}'.format(path))

def save_communites(partitions,path):
    '''
    保存社区到文件
    :param communities: 社区集合。
    :param out_filename: 输出文件名。
    '''
    with open(path,'w') as output_file:
    # output_file = open(path, 'w')
        for cmu in partitions:
            for member in cmu:
                output_file.write(str(member) + " ")
            output_file.write("\n")
        output_file.close()
        print('communities are stored to {}'.format(path))

def read_communities(path,use_set = False,n_add=0, G = None):
    '''

    :param path: 文件路径
    :param use_set: 列表中是否以集合形式返回
    :param n_add: 结点社区下标是否从0开始 此时G非NONE
    :param G: 已经有G图，与n_add配合使用
    :return: [[1,2,3],[4,5,6],...] or [{1,2,3},{4,5,6},...]
    '''

    partition = []
    with open(path, 'r') as f:
        for row in f.readlines():
            row = row.strip()

            r = row.strip().split('\t',-1)
            if r[0].find(' ')!=-1:
                r = r[0].split(' ',-1)

            if G is not None:
                r_filt = []
                for i in r:
                    nid = int(i) + n_add
                    if nid in G:
                        r_filt.append(nid)
            else:
                r_filt = list(map(lambda x:int(x)+n_add,r))

            if len(r_filt) == 0: continue
            # if len(r_filt) == 1: continue

            if use_set:
                partition.append(set(r_filt))
            else:
                partition.append(r_filt)

    # print(len(partition))
    return partition

def line_comms_to_node_comms(path1,path2):
    partition = []
    with open(path1, 'r') as f:
        for row in f.readlines():
            row = row.strip()

            r = row.strip().split('\t',-1)
            if r[0].find(' ')!=-1:
                r = r[0].split(' ',-1)

            r = list(map(int,r))
            partition.append(sorted(r,key=lambda x:x))

    output_file = open(path2, 'w')
    id = 1
    for cmu in partition:
        for member in cmu:
            output_file.write(str(member) + "\t" + str(id))
            output_file.write("\n")
        id += 1
    output_file.close()
    # print('communities are stored to {}'.format(path))


if __name__ == '__main__':
    p1 = r'J:\community detection\evolution\MODPSO\MODPSO\LFR_1000_20_50_20_100\community1k_0.6.txt'
    p2 = r'J:\community detection\evolution\MODPSO\MODPSO\LFR_1000_20_50_20_100\real_1k_0.6.txt'

    line_comms_to_node_comms(p1,p2)