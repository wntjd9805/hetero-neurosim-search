import argparse
import csv
import os
import numpy as np
from sys import path
path.append("/root/TOPSIS-Python/")
from topsis import Topsis

parser = argparse.ArgumentParser()
parser.add_argument("--model",  required=True)

args = parser.parse_args()



def simple_cull(inputPoints, dominates):
    paretoPoints = set()
    candidateRowNr = 0
    dominatedPoints = set()
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            if dominates(candidateRow, row):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow):
                nonDominated = False
                dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0:
            break
    return paretoPoints, dominatedPoints

def dominates(row, candidateRow):
  return sum([float(row[x]) <= float(candidateRow[x])*0.95 for x in range(len(row))]) == len(row)    


sa_set = []
pe_set = 0
tile_set = 0

with open(f"./search_space.txt") as f:
    lines = f.readlines()
    a = lines[0].split("=")[1].strip().split(',')
    for i in a:
        sa_set.append(int(i))
    pe_set = int(lines[1].split("=")[1].strip())
    tile_set = int(lines[2].split("=")[1].strip())
   
CONFIG = []
for sa_row in sa_set:
  for sa_col in sa_set:
    for pe in range(2,pe_set+1):
      for tile in range(4,tile_set+1):
        if tile > pe and tile%pe==0:
            if sa_row * pe >= sa_col and sa_row *tile >= sa_col and sa_col * pe >= sa_row and sa_col * tile >= sa_row:
                CONFIG.append([sa_row,sa_col, pe, tile])

config_homo = []
list_homo=[]
for config in CONFIG:
  NUM_ROW = config[0]
  NUM_COL = config[1]
  PE = config[2]
  Tile = config[3]

  fname = f"/root/hetero-neurosim-search/Inference_pytorch/search_result/{args.model}_homo/final_LATENCY_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}.txt"
  if os.path.isfile(fname):
    config_homo.append(f'SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}')
    with open(fname) as f:
        lines = f.readlines()
        print(lines)
        print(lines[0].split("\n")[0].split(","))
        list_homo.append(lines[0].split("\n")[0].split(","))
        

x = np.array([])
y = np.array([])
z = np.array([])
labels = np.array([])

for i,r in enumerate(list_homo):

  # if float(r[2]) < 43306397 and float(r[1]) < 121.74:  
  x=np.append(x,r[0])
  y=np.append(y,r[1])
  z=np.append(z,r[2])
  labels=np.append(labels,config_homo[i]) 
point = [list(element) for element in zip(x,y,z)]

pareto_label={}


for i,e in enumerate(point):
  if str(e[0])+str(e[1])+str(e[2]) in pareto_label.keys():
    pareto_label[str(e[0])+str(e[1])+str(e[2])].append(labels[i])
  else:
    pareto_label[str(e[0])+str(e[1])+str(e[2])]= [labels[i]]


paretoPoints, dominatedPoints = simple_cull(point, dominates)



paretoPoints_list=[]
CONFIG_pareto = []
for pareto in paretoPoints:
  # if float(pareto[1]) <319.2380155*0.8: #power
  #   if float(pareto[2]) < 88263661.6* 1.1: #area
    # if float(pareto[0]) <65808559.5* 1.1: #latency
        tmp_pareto =[pareto[0],pareto[1],pareto[2]]
        paretoPoints_list.append(tmp_pareto)
        CONFIG_pareto.append(pareto_label[str(pareto[0])+str(pareto[1])+str(pareto[2])])

# print(paretoPoints_list)
w = [1,1,10]
sign = np.array([False,False,False])
t = Topsis(paretoPoints_list, w, sign)
t.calc()



print("best_similarity\t", t.best_similarity[t.rank_to_best_similarity()[0]-1])

for i in range(10):
  print(f"--------{i}--------")
  print(float(paretoPoints_list[t.rank_to_best_similarity()[i]-1][0]),float(paretoPoints_list[t.rank_to_best_similarity()[i]-1][1]),float(paretoPoints_list[t.rank_to_best_similarity()[i]-1][2]))
  print(CONFIG_pareto[t.rank_to_best_similarity()[i]-1])