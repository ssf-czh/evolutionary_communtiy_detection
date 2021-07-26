
import collections
import os
import networkx as nx
import sys

# paths = [
         # ["./data/community.dat",
          # "./output/real.txt"],
         # ]


def formater(path):
    cmus = collections.defaultdict(lambda: list())
    with open(path[0], 'r') as f:
        for row in f.readlines():
            row = row.strip()
            # row = row.replace(" ","")
            r = row.split("\t",-1)
            if len(r) != 2:
                raise Exception("fail to decode community file, format error.")
            coms = r[1].split(" ",-1)
            for i in coms:
                cmus[i].append(r[0])
    # exit()
    # print(len(cmus))
    with open(path[1],"w+") as f:
        for key,item in cmus.items():
            line = " ".join(str(i) for i in sorted(item))+"\n"
            f.write(line)
    print(path[0]+" transform format to "+path[1]+"...Done")


def interface(input_path,output_path):
    separator_idx = input_path.rfind('.')
    file_name = input_path[:separator_idx]
    suffix = input_path[separator_idx+1:]

    if suffix == 'txt':
        print('suffix of input file can not be (.txt)')
        exit()
    formater([input_path, output_path])

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('cmd: python formater.py path')
        exit()
        
    path1 = sys.argv[1]
    separator_idx = path1.rfind('.')
    file_name = path1[:separator_idx]
    suffix = path1[separator_idx+1:]
    
    if suffix == 'txt':
        print('suffix of input file can not be (.txt)')
        exit()
    
    path2 = file_name + '.txt'
    # print(file_name,suffix)
    # print(path2)
    formater([path1,path2])
    
# for p in paths:
    # formater(p)