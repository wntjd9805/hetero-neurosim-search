import argparse
import csv
import os
import numpy as np
from sys import path
path.append("/root/TOPSIS-Python/")
from topsis import Topsis
import math


a = math.inf
parser = argparse.ArgumentParser()
parser.add_argument("--model",  required=True)
parser.add_argument('--distribute',type=int ,default=1, help='distribute')
parser.add_argument('--beam_size_m',type=int ,default=500,help='beam_size_m')
parser.add_argument('--beam_size_n',type=int ,default=3,help='beam_size_n')
parser.add_argument('--weight_latency',type=int ,default=0,required=True)
parser.add_argument('--weight_power',type=int ,default=0,required=True)
parser.add_argument('--weight_area',type=int ,default=0,required=True)
parser.add_argument('--constrain_latency',type=int ,default=a)
parser.add_argument('--constrain_power',type=int ,default=a)
parser.add_argument('--constrain_area',type=int ,default=a)
args = parser.parse_args()



CONFIG_pareto = []
paretoPoints_list=[]


fname = f"/root/hetero-neurosim-search/Inference_pytorch/search_result/{args.model}_hetero/final_result_{args.beam_size_m}_{args.beam_size_n}_{args.distribute}_[{args.weight_latency},{args.weight_power},{args.weight_area}].txt"

with open(fname) as f:
    lines = f.readlines()
    for l in lines:
        if float(l.split("\n")[0].split("\"")[2].split(",")[1:][0])<args.constrain_latency: #latency
            if float(l.split("\n")[0].split("\"")[2].split(",")[1:][1]) <args.constrain_power:  #power
                if float(l.split("\n")[0].split("\"")[2].split(",")[1:][2]) < args.constrain_area: #area
                    paretoPoints_list.append(l.split("\n")[0].split("\"")[2].split(",")[1:])
                    CONFIG_pareto.append(l.split("\n")[0].split("\"")[1])




w = [0, 1, 1]
sign = np.array([False,False,False])
t = Topsis(paretoPoints_list, w, sign)
t.calc()
print("best_similarity\t", t.best_similarity[t.rank_to_best_similarity()[0]-1])
for i in range(10):
    print(f"--------{i}--------")
    print(float(paretoPoints_list[t.rank_to_best_similarity()[i]-1][0]),float(paretoPoints_list[t.rank_to_best_similarity()[i]-1][1]),float(paretoPoints_list[t.rank_to_best_similarity()[i]-1][2]))
    print(CONFIG_pareto[t.rank_to_best_similarity()[i]-1])



