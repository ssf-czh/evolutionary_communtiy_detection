#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   gmlReader.py
@Contact :
@License :

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/8/19 0:32   linsher      1.0         None
'''
from collections import defaultdict
import networkx as nx
from tools import file_io


def parse_graph(path,id_key='id'):
    G = nx.read_gml(path, label=id_key)
    g_hash = dict()
    for n in G.nodes():
        g_hash[n] = 1
    return G,g_hash

def parse_comm(path,id_key = 'id',comm_key = 'value', g_hash=None):
    # off_count = 0
    prev_id = None
    comms = defaultdict(list)
    with open (path,'r') as f:
        for row in f.readlines():
            # labels have "xx"
            if -1 != row.find('"'):
                continue
            mkey = row.find(id_key)
            ckey = row.find(comm_key)
            if mkey != -1:
                s = row[mkey:]
                m_id = s[s.find(' '):s.find('\n')].replace(' ','')
                m_id = int(m_id)
                prev_id = m_id

            if ckey != -1:
                s = row[ckey:]
                c_id = s[s.find(' '):s.find('\n')].replace(' ','')
                c_id = int(c_id)
                if c_id in comms:
                    if g_hash is not None:
                        if prev_id in g_hash:
                            comms[c_id].append(prev_id)
                    else:
                        comms[c_id].append(prev_id)
                else:
                    if g_hash is not None:

                        if prev_id in g_hash:
                            comms[c_id] = [prev_id]
                    else:
                        comms[c_id] = [prev_id]

    comms_list = []
    for k,v in comms.items():
        comms_list.append(v)

    return comms_list




def read_gml(path,output_graph,output_comms):
    G , gHash =parse_graph(path,'id')
    comms = parse_comm(path,'id','value',g_hash = gHash)
    # print(comms)
    nx.write_edgelist(G,output_graph,data=False)
    file_io.save_communites(comms, output_comms)
    return comms

if __name__ == '__main__':
    # data = ['adjnoun',]
    p = r'D:\Chrome Download\cmdata2\数据\adjnoun.gml'

# p = r'D:\Chrome Download\cmdata2\数据\adjnoun.gml'
    p2=  '../dataset/real_data/adjnoun.txt'
    p3=  '../label/real_data/ans_adjnoun.txt'
    read_gml(p,output_graph=p2,output_comms=p3)
# gg  = nx.read_edgelist(p2,nodetype=int)

