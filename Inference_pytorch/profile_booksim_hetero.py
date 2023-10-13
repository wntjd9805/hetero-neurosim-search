import argparse
from audioop import ratecv
import os
from sqlite3 import Row
import time
from utee import misc
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utee import make_path
from utee import wage_util
# from models import dataset
import torchvision.models as models
import shutil
import sys
import multiprocessing
#from IPython import embed
from datetime import datetime
from subprocess import call

import tvm 
import tvm.relay as relay
import onnx
from parse import *
import re
import csv
import math
from itertools import combinations
import numpy as np
import ctypes
import copy
import subprocess
import random
import math
from multiprocessing import Pool,Manager
import itertools

from sys import path
path.append("/root/TOPSIS-Python/")
from topsis import Topsis

parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--model', default='ResNet50', help='VGG16|ResNet50|NasNetA|LFFD')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size for training (default: 64)')
parser.add_argument('--distribute',type=int ,default=1, help='distribute')
parser.add_argument('--beam_size_m',type=int ,default=500,help='beam_size_m')
parser.add_argument('--beam_size_n',type=int ,default=3,help='beam_size_n')
parser.add_argument('--latency',type=int ,default=1,help='weight_latency_with_pareto')
parser.add_argument('--power',type=int ,default=1,help='weight_power_with_pareto')
parser.add_argument('--area',type=int ,default=1,help='weight_area_with_pareto')
args = parser.parse_args()



def injection_rate(activation_size, Nbit, FPS, bus_width, freq):
    rate = (activation_size * Nbit * FPS) / (bus_width * freq)
    return rate


def comb2(arr):
    result = []
    for i in range(len(arr)):
        for j in arr[i + 1:]:
            result.append((arr[i], j))
    return result

def lcm(a,b):
  return (a * b) // math.gcd(a,b)

def flit_cycle(unit_rep,unit_wire,tile_width,clk,minDist):
  numRepeater = math.ceil(tile_width/minDist)
  if numRepeater>0:
    return math.ceil((unit_rep) * tile_width * clk)
  else:
    return math.ceil((unit_wire) * tile_width * clk)

def get_divisor(n ,minimum):
  data = []
  for i in range(1, n // 2 + 1):
      if n % i == 0:
          if i>=minimum:
            data.append(i)
  data.append(n)
  return data




def distribute(row,col,data_in,data_out, CLUSTER):
  data_intra_spread = data_in/row
  data_intra_gather = data_out/col
  # print("data_in",data_in)
  # print("data_out",data_out)
  minimum_split_row=math.ceil(row/CLUSTER)
  minimum_split_col=math.ceil(col/CLUSTER)
  candidate_row= get_divisor(row, minimum_split_row)
  candidate_col= get_divisor(col, minimum_split_col)
  result= None
  final_cost=None
  for split_row in candidate_row:
    for split_col in candidate_col:
      num_of_tile_in_cluster = (row/split_row) * (col/split_col)
      cost = data_in*split_col+(data_intra_gather+data_intra_spread)*num_of_tile_in_cluster+data_out*split_row
      if final_cost is None or cost < final_cost:
         final_cost = cost
         result = [split_row,split_col]
  # print("distribute")
  # print(result)
  return result
         
   

##place and rounting
def profile_for_select(i , tmp_dfg, Nbit ,Bus_width, chip_clk_freq, cluster_clk_freq, FPS ,CLUSTER, iter, big_wr,small_wr, shape, level, count, chip_width,chip_number,tile_grid, mapping_info):
  row = int(shape[2])
  col = int(shape[3])

  num_of_row = math.ceil(row/int(CLUSTER))
  num_of_col = math.ceil(col/int(CLUSTER))
  number_of_cluster = num_of_row * num_of_col

  key_list=list(tmp_dfg.keys())
  numBitToLoadIn=None
  numBitToLoadOut=None
  numInVector = None
  before_node_list =[]
  
  kernel_size = 1
  stride = 1
  if tmp_dfg[str(i)][6] != "":
    kernel_size = int(tmp_dfg[str(i)][6])
  if tmp_dfg[str(i)][7] != "":
    stride= int(tmp_dfg[str(i)][7])
    
  For_dense = None
  if i=='0':
    numInVector = (224-kernel_size+1)/stride * (224-kernel_size+1)/stride
    if args.model == "VGG8" or args.model == "DenseNet40":
      numInVector = (32-kernel_size+1)/stride * (32-kernel_size+1)/stride
    numBitToLoadIn = numInVector * Nbit *  kernel_size* kernel_size* 3
  else:
      for key_iter_for_fing_dest,p in enumerate(tmp_dfg.keys()):
          if tmp_dfg[p][1]==i or tmp_dfg[p][2]==i:
            if tmp_dfg[str(i)][0] =="nn.dense":
              For_dense = int(tmp_dfg[p][3][1])
            else:
              numInVector = (int(tmp_dfg[p][3][2])-kernel_size+1)/stride * (int(tmp_dfg[p][3][3])-kernel_size+1)/stride
              numBitToLoadIn = numInVector * Nbit * kernel_size* kernel_size *int(tmp_dfg[p][3][1])
            before_node_list.append(int(p))
              
  
  if tmp_dfg[str(i)][0] =="nn.dense":
    numInVector = 1
    numBitToLoadIn = numInVector * Nbit * For_dense
  injection = injection_rate(numBitToLoadIn/num_of_row, Nbit, FPS, Bus_width, cluster_clk_freq)

  numBitToLoadOut = numInVector * Nbit * int(tmp_dfg[str(i)][3][1])
  tmp_dfg[str(i)][4] = numBitToLoadOut
  
  
  for r in range(number_of_cluster):
    exist = 0
    for t in mapping_info.keys():
      if (i not in mapping_info[t][0]) and (mapping_info[t][1] == chip_number) and (mapping_info[t][3] >= int(row/math.ceil(row/CLUSTER))*int(col/math.ceil(col/CLUSTER))):
        mapping_info[t][0].append(i)
        exist = 1
        target= t.split("-")
        offset =int(CLUSTER**2)- mapping_info[t][3]
        node_in_tile =  mapping_info[t][2]
        break
    if exist == 0:
      tile_grid[iter[0],iter[1]] = i
      if (level+1)*(chip_width) <= count:
        iter[1]+=1
        level+=1
      else:
        if level%2==0:
          iter[0]+=1
        else:
          iter[0]-=1
      target= iter
      offset = 0
      node_in_tile = np.full(int(CLUSTER**2),-1)
      count+=1
   
    
    big_wr.writerow([i,tmp_dfg[str(i)][1],tmp_dfg[str(i)][2],tmp_dfg[str(i)][0],f"{target[0]}-{target[1]}",f"chip{chip_number}",numBitToLoadIn/num_of_row,injection,num_of_row,num_of_col])
   
    name_of_small_tile=f"{i}_{r}"
    small_tile1=np.full(int(CLUSTER**2),-1)

    for small in range(int(row/math.ceil(row/CLUSTER))*int(col/math.ceil(col/CLUSTER))):
      small_tile1[offset+small]=offset+small
      node_in_tile[offset+small]=i
    
    tmp_str1 = ""
    for li in small_tile1:
      tmp_str1= tmp_str1 +str(li) +" "

    injection_small_send = injection_rate(numBitToLoadIn/row, Nbit, FPS, Bus_width, chip_clk_freq)
    injection_small_receive = injection_rate(numBitToLoadOut/col, Nbit, FPS, Bus_width, chip_clk_freq)
    small_wr.writerow([name_of_small_tile,tmp_str1,(numBitToLoadIn/row),injection_small_send,(numBitToLoadOut/col),injection_small_receive,row,col])
    
    
    if exist == 1:
      mapping_info[f"{target[0]}-{target[1]}"][2] = node_in_tile
      mapping_info[f"{target[0]}-{target[1]}"][3] = mapping_info[f"{target[0]}-{target[1]}"][3] - int(row/math.ceil(row/CLUSTER))*int(col/math.ceil(col/CLUSTER))
    else:
      mapping_info[f"{target[0]}-{target[1]}"]=[[i],chip_number,node_in_tile,(CLUSTER**2)-int(row/math.ceil(row/CLUSTER))*int(col/math.ceil(col/CLUSTER))]
  return iter, count, level, tile_grid, num_of_col
  #----

def profile_for_select_distribute(i , tmp_dfg, Nbit ,Bus_width, chip_clk_freq, cluster_clk_freq, FPS ,CLUSTER, iter, big_wr,small_wr, shape, level, count, chip_width,chip_number,tile_grid, mapping_info):
  row = int(shape[2])
  col = int(shape[3])
  numBitToLoadIn=None
  numBitToLoadOut=None
  numInVector = None
  key_list=list(tmp_dfg.keys())
  before_node_list =[]


  kernel_size = 1
  stride = 1
  if tmp_dfg[str(i)][6] != "":
    kernel_size = int(tmp_dfg[str(i)][6])
  if tmp_dfg[str(i)][7] != "":
    stride= int(tmp_dfg[str(i)][7])
    
  For_dense = None
  if i=='0':
    numInVector = (224-kernel_size+1)/stride * (224-kernel_size+1)/stride
    if args.model == "VGG8" or args.model == "DenseNet40":
      numInVector = (32-kernel_size+1)/stride * (32-kernel_size+1)/stride
    numBitToLoadIn = numInVector * Nbit *  kernel_size* kernel_size* 3
  else:
    for key_iter_for_fing_dest,p in enumerate(tmp_dfg.keys()):
      if tmp_dfg[p][1]==i or tmp_dfg[p][2]==i:
        if tmp_dfg[str(i)][0] =="nn.dense":
          For_dense = int(tmp_dfg[p][3][1])
        else:
          numInVector = (int(tmp_dfg[p][3][2])-kernel_size+1)/stride * (int(tmp_dfg[p][3][3])-kernel_size+1)/stride
          numBitToLoadIn = numInVector * Nbit * kernel_size* kernel_size *int(tmp_dfg[p][3][1])
        before_node_list.append(int(p))
  
  if tmp_dfg[str(i)][0] =='nn.dense':
    numInVector = 1
    numBitToLoadIn = numInVector * Nbit * For_dense
    
  numBitToLoadOut = numInVector * Nbit * int(tmp_dfg[str(i)][3][1])
  tmp_dfg[str(i)][4] = numBitToLoadOut
  
  split = distribute(row,col,numBitToLoadIn,numBitToLoadOut,CLUSTER)

  num_of_row = split[0]
  num_of_col = split[1]
  number_of_cluster = num_of_row * num_of_col
  num_tile_per_cluster = (row/num_of_row) *(col/num_of_col)
  
  
  injection = injection_rate(numBitToLoadIn/num_of_row, Nbit, FPS, Bus_width, cluster_clk_freq)
  for r in range(number_of_cluster):
    exist = 0
    for t in mapping_info.keys():
      if (i not in mapping_info[t][0]) and (mapping_info[t][1] == chip_number) and (mapping_info[t][3] >= int(num_tile_per_cluster)):
        mapping_info[t][0].append(i)
        exist = 1
        target= t.split("-")
        offset =int(CLUSTER**2)- mapping_info[t][3]
        node_in_tile =  mapping_info[t][2]
        break
    if exist == 0:
      if iter[1]> chip_width:
        print(f"error_{mapping_info}_{chip_width}") 
      tile_grid[iter[0],iter[1]] = i
      if (level+1)*(chip_width) <= count:
        iter[1]+=1
        level+=1
      else:
        if level%2==0:
          iter[0]+=1
        else:
          iter[0]-=1
      count+=1
      target= iter
      offset = 0
      node_in_tile = np.full(int(CLUSTER**2),-1)
    
    
    big_wr.writerow([i,tmp_dfg[str(i)][1],tmp_dfg[str(i)][2],tmp_dfg[str(i)][0],f"{target[0]}-{target[1]}",f"chip{chip_number}",numBitToLoadIn/num_of_row,injection,num_of_row,num_of_col])

    name_of_small_tile=f"{i}_{r}"
    small_tile1=np.full(int(CLUSTER**2),-1)
    
    for small in range(int(num_tile_per_cluster)):
      small_tile1[offset+small]=offset+small
      node_in_tile[offset+small]=i
      
    tmp_str1 = ""
    for li in small_tile1:
      tmp_str1= tmp_str1 +str(li) +" "

    injection_small_send = injection_rate(numBitToLoadIn/row, Nbit, FPS, Bus_width, chip_clk_freq)
    injection_small_receive = injection_rate(numBitToLoadOut/col, Nbit, FPS, Bus_width, chip_clk_freq)
    small_wr.writerow([name_of_small_tile,tmp_str1,(numBitToLoadIn/row),injection_small_send,(numBitToLoadOut/col),injection_small_receive,row,col])
  
    if exist == 1:
      mapping_info[f"{target[0]}-{target[1]}"][2] = node_in_tile
      mapping_info[f"{target[0]}-{target[1]}"][3] = mapping_info[f"{target[0]}-{target[1]}"][3] - int(num_tile_per_cluster)
    else:
      mapping_info[f"{target[0]}-{target[1]}"]=[[i],chip_number,node_in_tile,(CLUSTER**2)-int(num_tile_per_cluster)]
  return iter, count, level, tile_grid, num_of_col
    

  
def number_of_tile_sofar(mapping_info):
  num_chip1=0
  num_chip2=0
  # print(mapping_info)
  for t in mapping_info.values():
    # print(t)
    if t[1] == 1:
      num_chip1+=1
    elif t[1] == 2:
      num_chip2 +=1
  return num_chip1,num_chip2
  
def PPA_function(latency, energy, area):
  return latency+energy/latency+area

def execute_booksim(node, cluster_width, chip1_width, chip2_width, cluster_flit_cycle, chip1_flit_cycle, chip2_flit_cycle,model,chip1,chip2,select, cluster_meter, chip1_meter,chip2_meter, chip_period,cluster_period, cluster_buswidth ,chip1_buswidth, chip2_buswidth):
  cmd1 = f'/root/hetero-neurosim-search/booksim2/src/booksim /root/hetero-neurosim-search/booksim2/src/examples/mesh88_lat {cluster_width} {chip1_width} {chip2_width} {cluster_flit_cycle} {chip1_flit_cycle} {chip2_flit_cycle} /root/hetero-neurosim-search/Inference_pytorch/record_{model}/CLUSTER/{depth}/CLUSTER_{model}_{chip1}_{chip2}{select}.txt /root/hetero-neurosim-search/Inference_pytorch/record_{model}/SMALL/{depth}/SMALL_{model}_{chip1}_{chip2}{select}.txt {cluster_meter} {chip1_meter} {chip2_meter} {cluster_buswidth} {chip1_buswidth} {chip2_buswidth} 0 na {node} 1 | egrep "taken|Total Power|Total Area|Total leak Power" > ./record_{args.model}/BOOKSIM/{depth}/BOOKSIM_{model}_{chip1}_{chip2}{select}.txt'
  cmd2 = f'/root/hetero-neurosim-search/booksim2/src/booksim /root/hetero-neurosim-search/booksim2/src/examples/mesh88_lat {cluster_width} {chip1_width} {chip2_width} {cluster_flit_cycle} {chip1_flit_cycle} {chip2_flit_cycle} /root/hetero-neurosim-search/Inference_pytorch/record_{model}/CLUSTER/{depth}/CLUSTER_{model}_{chip1}_{chip2}{select}.txt /root/hetero-neurosim-search/Inference_pytorch/record_{model}/SMALL/{depth}/SMALL_{model}_{chip1}_{chip2}{select}.txt {cluster_meter} {chip1_meter} {chip2_meter} {cluster_buswidth} {chip1_buswidth} {chip2_buswidth} 0 na {node} 2 | egrep "taken|Total Power|Total Area|Total leak Power" >> ./record_{args.model}/BOOKSIM/{depth}/BOOKSIM_{model}_{chip1}_{chip2}{select}.txt'
  cmd3 = f'/root/hetero-neurosim-search/booksim2/src/booksim /root/hetero-neurosim-search/booksim2/src/examples/mesh88_lat {cluster_width} {chip1_width} {chip2_width} {cluster_flit_cycle} {chip1_flit_cycle} {chip2_flit_cycle} /root/hetero-neurosim-search/Inference_pytorch/record_{model}/CLUSTER/{depth}/CLUSTER_{model}_{chip1}_{chip2}{select}.txt /root/hetero-neurosim-search/Inference_pytorch/record_{model}/SMALL/{depth}/SMALL_{model}_{chip1}_{chip2}{select}.txt {cluster_meter} {chip1_meter} {chip2_meter} {cluster_buswidth} {chip1_buswidth} {chip2_buswidth} 0 na {node} 3 | egrep "taken|Total Power|Total Area|Total leak Power" >> ./record_{args.model}/BOOKSIM/{depth}/BOOKSIM_{model}_{chip1}_{chip2}{select}.txt'
 

  try:
    output = subprocess.check_output(
        cmd1, stderr=subprocess.STDOUT, shell=True)
    
  except subprocess.CalledProcessError as exc:
      print("Error!!!!!", exc.returncode, exc.output, cmd1)
      return 10000000000000000,10000000000000000,100000000000000000,100000000000000000,100000000000000000,100000000000000000

  try:
    output = subprocess.check_output(
        cmd2, stderr=subprocess.STDOUT, shell=True)
  except subprocess.CalledProcessError as exc:
      print("Error!!!!!", exc.returncode, exc.output, cmd2)
      return 10000000000000000,10000000000000000,100000000000000000,100000000000000000,100000000000000000,100000000000000000

  try:
    output = subprocess.check_output(
        cmd3, stderr=subprocess.STDOUT, shell=True)
  except subprocess.CalledProcessError as exc:
      print("Error!!!!!", exc.returncode, exc.output, cmd3)
      return 10000000000000000,10000000000000000,100000000000000000,100000000000000000,100000000000000000,100000000000000000
  
  
  fname = f"./record_{args.model}/BOOKSIM/{depth}/BOOKSIM_{model}_{chip1}_{chip2}{select}.txt"
  latency_result = 0
  energy_result=0
  area_cluster=0
  area_chip = 0 
  cluster_leak_power = 0
  chip_leak_power = 0
  with open(fname) as f:
    lines = f.readlines()
    latency_result = int(lines[0].split("\n")[0].split(" ")[3])* cluster_period + int(lines[4].split("\n")[0].split(" ")[3])*chip_period + int(lines[8].split("\n")[0].split(" ")[3])*chip_period
    energy_result = float(lines[1].split("\n")[0].split(" ")[3])* int(lines[0].split("\n")[0].split(" ")[3])* cluster_period+ float(lines[5].split("\n")[0].split(" ")[3])*int(lines[4].split("\n")[0].split(" ")[3])*chip_period  + float(lines[9].split("\n")[0].split(" ")[3])*int(lines[8].split("\n")[0].split(" ")[3])*chip_period
    area_cluster = float(lines[2].split("\n")[0].split(" ")[3])*1000000
    area_chip = float(lines[6].split("\n")[0].split(" ")[3])*1000000
    cluster_leak_power = float(lines[3].split("\n")[0].split(" ")[4])
    chip_leak_power = float(lines[7].split("\n")[0].split(" ")[4])

  # print(cluster_period)
  # print("energy_result",energy_result*1000)
  # print(area_cluster)
  # print(cluster_leak_power)
  # print(chip_leak_power)
  
  # print("CLUSTER: ", int(lines[0].split("\n")[0].split(" ")[3]) ,"cycles")
  # print("Send: ", int(lines[1].split("\n")[0].split(" ")[3]) ,"cycles")
  # print("Receive: ", int(lines[2].split("\n")[0].split(" ")[3]) ,"cycles")

  # print(node,cycle)
  # print(node,latency_result)
  # time.sleep(0.5)
  return latency_result,energy_result*1000,area_cluster,area_chip,cluster_leak_power,chip_leak_power


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


def initialization(config):
  NUM1_ROW = config[0][0]
  NUM1_COL = config[0][1]
  PE1 = config[0][2]
  Tile1 = config[0][3]
  NUM2_ROW = config[1][0]
  NUM2_COL = config[1][1]
  PE2 = config[1][2]
  Tile2 = config[1][3]
  
  shape1={}
  shape2={}
  fname1 = f"./shape/shape_{args.model}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"
  fname2 = f"./shape/shape_{args.model}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}"
  with open(fname1) as f:
      lines = f.readlines()
      for i, l in enumerate(lines):
          l=l.replace("\n", "")
          shape1[i]=l.split(',')
  with open(fname2) as f:
      lines = f.readlines()
      for i, l in enumerate(lines):
          l=l.replace("\n", "")
          shape2[i]=l.split(',')
  
  CLUSTER1 = int(max(NUM1_ROW*Tile1,NUM2_ROW*Tile2)/(NUM1_ROW*Tile1))
  CLUSTER2 = int(max(NUM1_ROW*Tile1,NUM2_ROW*Tile2)/(NUM2_ROW*Tile2))
  
  compute_PPA1={}
  compute_PPA2={}
  FPS_latency = 0
  tot_cluster_tile=0

  for layer_num, layer in enumerate(All_layer):
    latency1 = LATENCY.get(f"{layer}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")-inter_connect_LATENCY.get(f"{layer}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
    latency2 = LATENCY.get(f"{layer}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}")-inter_connect_LATENCY.get(f"{layer}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}")
    energy1 = POWER.get(f"{layer}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")-inter_connect_POWER.get(f"{layer}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
    energy2 = POWER.get(f"{layer}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}")-inter_connect_POWER.get(f"{layer}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}")
    area1 = AREA.get(f"{layer}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")-inter_connect_AREA.get(f"{layer}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
    area2 = AREA.get(f"{layer}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}")-inter_connect_AREA.get(f"{layer}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}")
    compute_PPA1[layer_num] = [latency1,energy1,area1] 
    compute_PPA2[layer_num] = [latency2,energy2,area2]
    if compute_PPA1[layer_num][0] + energy1/10 > compute_PPA2[layer_num][0] + energy2/10:
      FPS_latency = FPS_latency + float(LATENCY.get(f"{layer}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}"))*1e-9
      
    else:
      FPS_latency = FPS_latency + float(LATENCY.get(f"{layer}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"))*1e-9

    num_cluster = max(math.ceil(int(shape1[layer_num][4]) / (CLUSTER1**2)), math.ceil(int(shape2[layer_num][4]) / (CLUSTER2**2)))
    tot_cluster_tile = tot_cluster_tile + num_cluster
  FPS = 1/FPS_latency
  if args.distribute == 1:
    chip_width = math.ceil(math.sqrt(tot_cluster_tile)) + 2
  else:
    chip_width = math.ceil(math.sqrt(tot_cluster_tile)) + 1
  X = np.linspace(-1,-1,chip_width)
  Y = np.linspace(-1,-1,chip_width)
  tile_grid,tile_type = np.meshgrid(X,Y)
  iter=[-1,1]
  level = np.array([0], dtype=int)
  count = np.array([0], dtype=int)
  tmp_dfg = dfg
  mapping_info = {}
  
  tail_distribute = ""
  if args.distribute == 1:
     tail_distribute = "distribute"
  
  big_f = open(f"./record_{args.model}/CLUSTER/{depth}/CLUSTER_{args.model}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}_{tail_distribute}.txt",'w', newline='')
  big_wr = csv.writer(big_f)
  big_wr.writerow(["node","destination1","destination2","op","location","type","activation_size","injection_rate"])
  big_f.close()
  small_f = open(f"./record_{args.model}/SMALL/{depth}/SMALL_{args.model}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}_{tail_distribute}.txt",'w', newline='')
  small_wr = csv.writer(small_f)
  small_wr.writerow(["node","used","activation_size","injection_rate"])
  small_f.close()
  conv_dense=0
  total_latency=0
  total_energy=0
  total_area=0
  leakage=0
  selected_list=[]
  
  
  return shape1,shape2,compute_PPA1,compute_PPA2,chip_width,FPS,iter,level,count,tile_grid,mapping_info,conv_dense,total_latency,total_energy,total_area,leakage,selected_list
  
def make_args(config,tmp_dfg,shape1,shape2,compute_PPA1,compute_PPA2,chip_width,FPS,iter,level,count,tile_grid,mapping_info,conv_dense,total_latency,total_energy,dfg_key,selected_list):
  NUM1_ROW = config[0][0]
  NUM1_COL = config[0][1]
  PE1 = config[0][2]
  Tile1 = config[0][3]
  NUM2_ROW = config[1][0]
  NUM2_COL = config[1][1]
  PE2 = config[1][2]
  Tile2 = config[1][3]
  CLUSTER1 = int(max(NUM1_ROW*Tile1,NUM2_ROW*Tile2)/(NUM1_ROW*Tile1))
  CLUSTER2 = int(max(NUM1_ROW*Tile1,NUM2_ROW*Tile2)/(NUM2_ROW*Tile2))

    

  chip1_flit_cycle = flit_cycle(unitLatencyRep.get(f"SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"), unitLatencyWire.get(f"SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"),tile_width_meter.get(f"SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"),clk_frequency.get(f"SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"),minDist.get(f"SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"))
  chip2_flit_cycle = flit_cycle(unitLatencyRep.get(f"SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}"), unitLatencyWire.get(f"SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}"),tile_width_meter.get(f"SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}"),clk_frequency.get(f"SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}"),minDist.get(f"SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}"))
  cluster_meter = max(tile_width_meter.get(f"SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")*CLUSTER1, tile_width_meter.get(f"SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}")*CLUSTER2)
  chip1_clk_freq = clk_frequency.get(f"SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
  chip2_clk_freq = clk_frequency.get(f"SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}")
  cluster_clk_freq = min(clk_frequency.get(f"SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"), clk_frequency.get(f"SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}"))
  cluster_flit_cycle = max(chip1_flit_cycle,chip2_flit_cycle)
  # flit_cycle(cluster_Rep, cluster_wire, cluster_meter, cluster_clk_freq)
  chip1_meter = tile_width_meter.get(f"SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
  chip2_meter = tile_width_meter.get(f"SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}")
  chip1_buswidth = busWidth.get(f"SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
  chip2_buswidth = busWidth.get(f"SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}")
  cluster_buswidth = max(chip1_buswidth,chip2_buswidth)
  chip1_period = clk_period.get(f"SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")
  chip2_period = clk_period.get(f"SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}")
  period = max(clk_period.get(f"SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}"), clk_period.get(f"SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}"))
  




  total_area = 0
  total_leakage =0
  chip1_area = chip1_meter**2*1e12
  chip2_area = chip2_meter**2*1e12
  chip1_leakage = leakage_POWER.get(f"layer1_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}")/int(shape1[1][4])*1e-6
  chip2_leakage = leakage_POWER.get(f"layer1_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}")/int(shape2[1][4])*1e-6
  chip1_booksim_area  = 0
  chip2_booksim_area = 0
  cluster_booksim_area =0
  chip1_booksim_leakage  = 0
  chip2_booksim_leakage = 0
  cluster_booksim_leakage = 0

  tail_selet=""
  for selected in selected_list:
    tail_selet = tail_selet + str(selected)+"_"
  
  
  tail_distribute = tail_selet
  if args.distribute == 1:
     tail_distribute = tail_selet+"distribute"

  Nbit=8
  node_col={}
  inverse = {}
  tail1=f"{tail_selet}1_"
  tail2=f"{tail_selet}2_"
  if args.distribute == 1:
    tail1=f"{tail_selet}1_distribute"
    tail2=f"{tail_selet}2_distribute"
    
    
  for i in dfg_key:
    # print(f"---------------------{i}---------------------")
    # print(tmp_dfg[str(i)][0])  
    if (tmp_dfg[str(i)][0]=='nn.conv2d' or tmp_dfg[str(i)][0]=='nn.dense'):
      shutil.copy(f"./record_{args.model}/CLUSTER/{depth-1}/CLUSTER_{args.model}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}_{tail_distribute}.txt" ,f"./record_{args.model}/CLUSTER/{depth}/CLUSTER_{args.model}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}_{tail1}.txt")
      shutil.copy(f"./record_{args.model}/CLUSTER/{depth-1}/CLUSTER_{args.model}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}_{tail_distribute}.txt" ,f"./record_{args.model}/CLUSTER/{depth}/CLUSTER_{args.model}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}_{tail2}.txt")
      shutil.copy(f"./record_{args.model}/SMALL/{depth-1}/SMALL_{args.model}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}_{tail_distribute}.txt" ,f"./record_{args.model}/SMALL/{depth}/SMALL_{args.model}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}_{tail1}.txt")
      shutil.copy(f"./record_{args.model}/SMALL/{depth-1}/SMALL_{args.model}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}_{tail_distribute}.txt", f"./record_{args.model}/SMALL/{depth}/SMALL_{args.model}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}_{tail2}.txt")
      iter1 = copy.deepcopy(iter)
      iter2 = copy.deepcopy(iter)
      level1 = copy.deepcopy(level)
      level2 = copy.deepcopy(level)
      count1 = copy.deepcopy(count)
      count2 = copy.deepcopy(count)
      tile_grid1 = copy.deepcopy(tile_grid)
      tile_grid2 = copy.deepcopy(tile_grid)
      mapping_info1 = copy.deepcopy(mapping_info)
      mapping_info2 = copy.deepcopy(mapping_info)
      selected_list1= copy.deepcopy(selected_list)
      selected_list2= copy.deepcopy(selected_list)
      copy_lock = 1

      big_f_select1 = open(f"./record_{args.model}/CLUSTER/{depth}/CLUSTER_{args.model}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}_{tail1}.txt",'a')
      big_wr_select1 = csv.writer(big_f_select1)
      big_f_select2 = open(f"./record_{args.model}/CLUSTER/{depth}/CLUSTER_{args.model}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}_{tail2}.txt",'a')
      big_wr_select2 = csv.writer(big_f_select2)
      small_f_select1 = open(f"./record_{args.model}/SMALL/{depth}/SMALL_{args.model}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}_{tail1}.txt",'a')
      small_wr_select1 = csv.writer(small_f_select1)
      small_f_select2 = open(f"./record_{args.model}/SMALL/{depth}/SMALL_{args.model}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}_{tail2}.txt",'a')
      small_wr_select2 = csv.writer(small_f_select2)
      num_of_col1 = None
      num_of_col2 = None
      if args.distribute == 1:
        iter1, count1, level1, tile_grid1,num_of_col1 = profile_for_select_distribute(i, tmp_dfg , Nbit ,chip1_buswidth, chip1_clk_freq, cluster_clk_freq, FPS ,CLUSTER1, iter1, big_wr_select1 ,small_wr_select1, shape1.get(conv_dense), level1, count1, chip_width, 1, tile_grid1,mapping_info1)
        iter2, count2, level2, tile_grid2,num_of_col2 = profile_for_select_distribute(i, tmp_dfg , Nbit ,chip2_buswidth, chip2_clk_freq, cluster_clk_freq, FPS ,CLUSTER2, iter2, big_wr_select2 ,small_wr_select2, shape2.get(conv_dense), level2, count2, chip_width, 2, tile_grid2,mapping_info2)
      else:
        iter1, count1, level1, tile_grid1,num_of_col1 = profile_for_select(i, tmp_dfg , Nbit ,chip1_buswidth, chip1_clk_freq, cluster_clk_freq, FPS ,CLUSTER1, iter1, big_wr_select1 ,small_wr_select1, shape1.get(conv_dense), level1, count1, chip_width, 1, tile_grid1,mapping_info1)
        iter2, count2, level2, tile_grid2,num_of_col2 = profile_for_select(i, tmp_dfg , Nbit ,chip2_buswidth, chip2_clk_freq, cluster_clk_freq, FPS ,CLUSTER2, iter2, big_wr_select2 ,small_wr_select2, shape2.get(conv_dense), level2, count2, chip_width, 2, tile_grid2,mapping_info2)
      # cmd1 = f'python booksim_select.py --model={args.model} --node={i} --select=_select1 --chip1=SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1} --chip2=SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2} --cluster_flit_cycle={cluster_flit_cycle} --chip1_flit_cycle={chip1_flit_cycle} --chip2_flit_cycle={chip2_flit_cycle} --cluster_width={chip_width} --chip1_width={int(CLUSTER1)} --chip2_width={int(CLUSTER2)} --cluster_meter={cluster_meter} --chip1_meter={chip1_meter} --chip2_meter={chip2_meter} --clk_period={period}'
      # cmd2 = f'python booksim_select.py --model={args.model} --node={i} --select=_select2 --chip1=SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1} --chip2=SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2} --cluster_flit_cycle={cluster_flit_cycle} --chip1_flit_cycle={chip1_flit_cycle} --chip2_flit_cycle={chip2_flit_cycle} --cluster_width={chip_width} --chip1_width={int(CLUSTER1)} --chip2_width={int(CLUSTER2)} --cluster_meter={cluster_meter} --chip1_meter={chip1_meter} --chip2_meter={chip2_meter} --clk_period={period}'
      # cmd1 = ['python', 'booksim_select.py', f'--model={args.model}', f'--node={i}', f'--select=_select1', f'--chip1=SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}', f'--chip2=SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}', f'--cluster_flit_cycle={cluster_flit_cycle}', f'--chip1_flit_cycle={chip1_flit_cycle}', f'--chip2_flit_cycle={chip2_flit_cycle}', f'--cluster_width={chip_width}', f'--chip1_width={int(CLUSTER1)}', f'--chip2_width={int(CLUSTER2)}', f'--cluster_meter={cluster_meter}', f'--chip1_meter={chip1_meter}', f'--chip2_meter={chip2_meter}', f'--clk_period={period}']
      # cmd2 = ['python', 'booksim_select.py', f'--model={args.model}', f'--node={i}', f'--select=_select2', f'--chip1=SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}', f'--chip2=SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}', f'--cluster_flit_cycle={cluster_flit_cycle}', f'--chip1_flit_cycle={chip1_flit_cycle}', f'--chip2_flit_cycle={chip2_flit_cycle}', f'--cluster_width={chip_width}', f'--chip1_width={int(CLUSTER1)}', f'--chip2_width={int(CLUSTER2)}', f'--cluster_meter={cluster_meter}', f'--chip1_meter={chip1_meter}', f'--chip2_meter={chip2_meter}', f'--clk_period={period}']
      big_f_select1.close()
      big_f_select2.close()
      small_f_select1.close()
      small_f_select2.close()
      # print(f'--------------layer{conv_dense}-----------------')
      # print(f'SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}')
      # print("compute latency", compute_PPA1[conv_dense])
      if depth==0 or depth ==1:
        if os.path.isfile(f"./record_{args.model}/Prepared_data/{depth}/CLUSTER_{args.model}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}_{i}.txt"):
          fname = f"./record_{args.model}/Prepared_data/{depth}/CLUSTER_{args.model}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}_{i}.txt"
          with open(fname) as f:
            lines = f.readlines()
            if len(lines) >0:
              booksim_latency1=float(lines[0].split(',')[0])
              booksim_energy1=float(lines[0].split(',')[1])
              cluster_booksim_area=float(lines[0].split(',')[2])
              chip1_booksim_area=float(lines[0].split(',')[3])
              cluster_booksim_leakage==float(lines[0].split(',')[4])
              chip1_booksim_leakage=float(lines[0].split(',')[5])
              
              booksim_latency2=float(lines[1].split(',')[0])
              booksim_energy2=float(lines[1].split(',')[1])
              cluster_booksim_area=float(lines[1].split(',')[2])
              chip2_booksim_area=float(lines[1].split(',')[3])
              cluster_booksim_leakage==float(lines[1].split(',')[4])
              chip2_booksim_leakage=float(lines[1].split(',')[5])
            else:
              booksim_latency1,booksim_energy1,cluster_booksim_area,chip1_booksim_area,cluster_booksim_leakage,chip1_booksim_leakage = execute_booksim(i,chip_width,int(CLUSTER1),int(CLUSTER2),cluster_flit_cycle,chip1_flit_cycle,chip2_flit_cycle,args.model,f'SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}',f'SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}',f"_{tail1}",cluster_meter,chip1_meter,chip2_meter,chip1_period,period,cluster_buswidth,chip1_buswidth,chip2_buswidth)
              booksim_latency2,booksim_energy2,cluster_booksim_area,chip2_booksim_area,cluster_booksim_leakage,chip2_booksim_leakage = execute_booksim(i,chip_width,int(CLUSTER1),int(CLUSTER2),cluster_flit_cycle,chip1_flit_cycle,chip2_flit_cycle,args.model,f'SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}',f'SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}',f"_{tail2}",cluster_meter,chip1_meter,chip2_meter,chip2_period,period,cluster_buswidth,chip1_buswidth,chip2_buswidth)
              prepared1 = open(f"./record_{args.model}/Prepared_data/{depth}/CLUSTER_{args.model}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}_{i}.txt",'w')
              wr_prepared1 = csv.writer(prepared1)
              wr_prepared1.writerow([booksim_latency1,booksim_energy1,cluster_booksim_area,chip1_booksim_area,cluster_booksim_leakage,chip1_booksim_leakage])
              wr_prepared1.writerow([booksim_latency2,booksim_energy2,cluster_booksim_area,chip2_booksim_area,cluster_booksim_leakage,chip2_booksim_leakage])


            
        else:
          booksim_latency1,booksim_energy1,cluster_booksim_area,chip1_booksim_area,cluster_booksim_leakage,chip1_booksim_leakage = execute_booksim(i,chip_width,int(CLUSTER1),int(CLUSTER2),cluster_flit_cycle,chip1_flit_cycle,chip2_flit_cycle,args.model,f'SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}',f'SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}',f"_{tail1}",cluster_meter,chip1_meter,chip2_meter,chip1_period,period,cluster_buswidth,chip1_buswidth,chip2_buswidth)
          booksim_latency2,booksim_energy2,cluster_booksim_area,chip2_booksim_area,cluster_booksim_leakage,chip2_booksim_leakage = execute_booksim(i,chip_width,int(CLUSTER1),int(CLUSTER2),cluster_flit_cycle,chip1_flit_cycle,chip2_flit_cycle,args.model,f'SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}',f'SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}',f"_{tail2}",cluster_meter,chip1_meter,chip2_meter,chip2_period,period,cluster_buswidth,chip1_buswidth,chip2_buswidth)
          prepared1 = open(f"./record_{args.model}/Prepared_data/{depth}/CLUSTER_{args.model}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}_{i}.txt",'w')
          wr_prepared1 = csv.writer(prepared1)
          wr_prepared1.writerow([booksim_latency1,booksim_energy1,cluster_booksim_area,chip1_booksim_area,cluster_booksim_leakage,chip1_booksim_leakage])
          wr_prepared1.writerow([booksim_latency2,booksim_energy2,cluster_booksim_area,chip2_booksim_area,cluster_booksim_leakage,chip2_booksim_leakage])
      else:
        booksim_latency1,booksim_energy1,cluster_booksim_area,chip1_booksim_area,cluster_booksim_leakage,chip1_booksim_leakage = execute_booksim(i,chip_width,int(CLUSTER1),int(CLUSTER2),cluster_flit_cycle,chip1_flit_cycle,chip2_flit_cycle,args.model,f'SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}',f'SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}',f"_{tail1}",cluster_meter,chip1_meter,chip2_meter,chip1_period,period,cluster_buswidth,chip1_buswidth,chip2_buswidth)
        booksim_latency2,booksim_energy2,cluster_booksim_area,chip2_booksim_area,cluster_booksim_leakage,chip2_booksim_leakage = execute_booksim(i,chip_width,int(CLUSTER1),int(CLUSTER2),cluster_flit_cycle,chip1_flit_cycle,chip2_flit_cycle,args.model,f'SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}',f'SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}',f"_{tail2}",cluster_meter,chip1_meter,chip2_meter,chip2_period,period,cluster_buswidth,chip1_buswidth,chip2_buswidth)
    
  
      inverse_key = tmp_dfg[str(i)][1]
      if tmp_dfg[str(i)][1] == '':
        inverse_key = tmp_dfg[str(i)][2]
      
      if inverse_key in inverse.keys():
        inverse[inverse_key][0][0] = inverse[inverse_key][0][0] + booksim_latency1 + compute_PPA1[conv_dense][0]
        inverse[inverse_key][0][1] = inverse[inverse_key][0][1] + booksim_energy1 + compute_PPA1[conv_dense][1]
        inverse[inverse_key][1][0] = inverse[inverse_key][1][0] + booksim_latency2 + compute_PPA2[conv_dense][0]
        inverse[inverse_key][1][0] = inverse[inverse_key][1][0] + booksim_energy2 + compute_PPA2[conv_dense][1]
      else:
        inverse[inverse_key] = [[booksim_latency1 + compute_PPA1[conv_dense][0], booksim_energy1 + compute_PPA1[conv_dense][1]], [booksim_latency2 + compute_PPA2[conv_dense][0], booksim_energy2 + compute_PPA2[conv_dense][1]]] 
      
      before_node = i

      if dfg_key[-1] == i:
        num_chip1_sofar1,num_chip2_sofar1 = number_of_tile_sofar(mapping_info1)
        total_latency1 = total_latency  + booksim_latency1 + compute_PPA1[conv_dense][0]
        total_energy1 = total_energy + booksim_energy1 + compute_PPA1[conv_dense][1]
        leakage1 = (cluster_booksim_leakage/(chip_width**2)*chip_width + (cluster_booksim_leakage/(chip_width**2) + chip1_booksim_leakage) * num_chip1_sofar1 + chip1_leakage*num_chip1_sofar1*(CLUSTER1**2) + (cluster_booksim_leakage/(chip_width**2)+chip2_booksim_leakage) * num_chip2_sofar1 + chip2_leakage*num_chip2_sofar1*(CLUSTER2**2) )* total_latency1 * 1000
        total_area1 = cluster_booksim_area + chip1_booksim_area* num_chip1_sofar1+ chip1_area*num_chip1_sofar1*(CLUSTER1**2) + chip2_booksim_area*num_chip2_sofar1+ chip2_area*num_chip2_sofar1*(CLUSTER2**2) 
        selected_list1.append(1)
        
        num_chip1_sofar2,num_chip2_sofar2 = number_of_tile_sofar(mapping_info2)
        total_latency2 = total_latency  + booksim_latency2 + compute_PPA2[conv_dense][0]
        total_energy2 = total_energy + booksim_energy2 + compute_PPA2[conv_dense][1]
        leakage2 = (cluster_booksim_leakage/(chip_width**2)*chip_width + (cluster_booksim_leakage/(chip_width**2) + chip1_booksim_leakage) * num_chip1_sofar2 + chip1_leakage*num_chip1_sofar2*(CLUSTER1**2) + (cluster_booksim_leakage/(chip_width**2)+chip2_booksim_leakage) * num_chip2_sofar2 + chip2_leakage*num_chip2_sofar2*(CLUSTER2**2) )* total_latency2 * 1000
        total_area2 = cluster_booksim_area + chip1_booksim_area* num_chip1_sofar2+ chip1_area*num_chip1_sofar2*(CLUSTER1**2) + chip2_booksim_area*num_chip2_sofar2+ chip2_area*num_chip2_sofar2*(CLUSTER2**2) 
        selected_list2.append(2)
        conv_dense=conv_dense+1
        return iter1,level1,count1,tile_grid1,mapping_info1,conv_dense,total_latency1,total_energy1,total_area1,leakage1,selected_list1, iter2,level2,count2,tile_grid2,mapping_info2,conv_dense,total_latency2,total_energy2,total_area2,leakage2,selected_list2

      conv_dense=conv_dense+1

    else:
      before_conv_result1 = inverse[i][0]
      before_conv_result2 = inverse[i][1]
      # print(before_conv_result1)
      # print(before_conv_result2)
      key_loc_before_size=None
      before_node_list=[]
      for key_iter_for_fing_dest,p in enumerate(tmp_dfg.keys()):
          if tmp_dfg[p][1]==i or tmp_dfg[p][2]==i:
            key_loc_before_size = tmp_dfg[p][4]
            before_node_list.append(int(p))


      injection1 = injection_rate(key_loc_before_size/num_of_col1, Nbit, FPS,  chip1_buswidth, cluster_clk_freq)
      injection2 = injection_rate(key_loc_before_size/num_of_col2, Nbit, FPS,  chip2_buswidth, cluster_clk_freq)
      big_f_select1 = open(f"./record_{args.model}/CLUSTER/{depth}/CLUSTER_{args.model}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}_{tail1}.txt",'a')
      big_wr_select1 = csv.writer(big_f_select1)
      big_wr_select1.writerow([i,tmp_dfg[str(i)][1],tmp_dfg[str(i)][2],tmp_dfg[str(i)][0],f"{iter1[0]}-0","non_MAC",key_loc_before_size/num_of_col1,injection1,1,1])
      big_f_select1.close()
      
      big_f_select2 = open(f"./record_{args.model}/CLUSTER/{depth}/CLUSTER_{args.model}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}_{tail2}.txt",'a')
      big_wr_select2 = csv.writer(big_f_select2)
      big_wr_select2.writerow([i,tmp_dfg[str(i)][1],tmp_dfg[str(i)][2],tmp_dfg[str(i)][0],f"{iter2[0]}-0","non_MAC",key_loc_before_size/num_of_col2,injection2,1,1])
      big_f_select2.close()
      if depth==0 or depth ==1:
        if os.path.isfile(f"./record_{args.model}/Prepared_data/{depth}/CLUSTER_{args.model}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}_{i}.txt"):
          fname = f"./record_{args.model}/Prepared_data/{depth}/CLUSTER_{args.model}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}_{i}.txt"
          with open(fname) as f:
            lines = f.readlines()
            if len(lines) >0:
              booksim_latency1=float(lines[0].split(',')[0])
              booksim_energy1=float(lines[0].split(',')[1])
              cluster_booksim_area=float(lines[0].split(',')[2])
              chip1_booksim_area=float(lines[0].split(',')[3])
              cluster_booksim_leakage==float(lines[0].split(',')[4])
              chip1_booksim_leakage=float(lines[0].split(',')[5])
              
              booksim_latency2=float(lines[1].split(',')[0])
              booksim_energy2=float(lines[1].split(',')[1])
              cluster_booksim_area=float(lines[1].split(',')[2])
              chip2_booksim_area=float(lines[1].split(',')[3])
              cluster_booksim_leakage==float(lines[1].split(',')[4])
              chip2_booksim_leakage=float(lines[1].split(',')[5])
              booksim_temp1=[booksim_latency1,booksim_energy1,cluster_booksim_area,chip1_booksim_area,cluster_booksim_leakage,chip1_booksim_leakage]
              booksim_temp2=[booksim_latency2,booksim_energy2,cluster_booksim_area,chip2_booksim_area,cluster_booksim_leakage,chip2_booksim_leakage]
            else:
              booksim_temp1= execute_booksim(i,chip_width,int(CLUSTER1),int(CLUSTER2),cluster_flit_cycle,chip1_flit_cycle,chip2_flit_cycle,args.model,f'SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}',f'SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}',f"_{tail1}",cluster_meter,chip1_meter,chip2_meter,period,period,cluster_buswidth,chip1_buswidth,chip2_buswidth)
              booksim_temp2= execute_booksim(i,chip_width,int(CLUSTER1),int(CLUSTER2),cluster_flit_cycle,chip1_flit_cycle,chip2_flit_cycle,args.model,f'SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}',f'SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}',f"_{tail2}",cluster_meter,chip1_meter,chip2_meter,period,period,cluster_buswidth,chip1_buswidth,chip2_buswidth)
              prepared1 = open(f"./record_{args.model}/Prepared_data/{depth}/CLUSTER_{args.model}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}_{i}.txt",'w')
              wr_prepared1 = csv.writer(prepared1)
              wr_prepared1.writerow(booksim_temp1)
              wr_prepared1.writerow(booksim_temp2)
        else:
          booksim_temp1= execute_booksim(i,chip_width,int(CLUSTER1),int(CLUSTER2),cluster_flit_cycle,chip1_flit_cycle,chip2_flit_cycle,args.model,f'SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}',f'SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}',f"_{tail1}",cluster_meter,chip1_meter,chip2_meter,period,period,cluster_buswidth,chip1_buswidth,chip2_buswidth)
          booksim_temp2= execute_booksim(i,chip_width,int(CLUSTER1),int(CLUSTER2),cluster_flit_cycle,chip1_flit_cycle,chip2_flit_cycle,args.model,f'SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}',f'SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}',f"_{tail2}",cluster_meter,chip1_meter,chip2_meter,period,period,cluster_buswidth,chip1_buswidth,chip2_buswidth)
          prepared1 = open(f"./record_{args.model}/Prepared_data/{depth}/CLUSTER_{args.model}_SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}_SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}_{i}.txt",'w')
          wr_prepared1 = csv.writer(prepared1)
          wr_prepared1.writerow(booksim_temp1)
          wr_prepared1.writerow(booksim_temp2)
      else:
        booksim_temp1= execute_booksim(i,chip_width,int(CLUSTER1),int(CLUSTER2),cluster_flit_cycle,chip1_flit_cycle,chip2_flit_cycle,args.model,f'SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}',f'SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}',f"_{tail1}",cluster_meter,chip1_meter,chip2_meter,period,period,cluster_buswidth,chip1_buswidth,chip2_buswidth)
        booksim_temp2= execute_booksim(i,chip_width,int(CLUSTER1),int(CLUSTER2),cluster_flit_cycle,chip1_flit_cycle,chip2_flit_cycle,args.model,f'SA_row:{NUM1_ROW}_SA_col:{NUM1_COL}_PE:{PE1}_TL:{Tile1}',f'SA_row:{NUM2_ROW}_SA_col:{NUM2_COL}_PE:{PE2}_TL:{Tile2}',f"_{tail2}",cluster_meter,chip1_meter,chip2_meter,period,period,cluster_buswidth,chip1_buswidth,chip2_buswidth)
      
      num_chip1_sofar1,num_chip2_sofar1 = number_of_tile_sofar(mapping_info1)
      total_latency1 = total_latency + booksim_temp1[0] + before_conv_result1[0]
      total_energy1 = total_energy + booksim_temp1[1] + before_conv_result1[1]
      leakage1 = (cluster_booksim_leakage/(chip_width**2)*chip_width + (cluster_booksim_leakage/(chip_width**2) + chip1_booksim_leakage) * num_chip1_sofar1 + chip1_leakage*num_chip1_sofar1*(CLUSTER1**2) + (cluster_booksim_leakage/(chip_width**2)+chip2_booksim_leakage) * num_chip2_sofar1 + chip2_leakage*num_chip2_sofar1*(CLUSTER2**2) )* total_latency1 * 1000
      total_area1 = cluster_booksim_area + chip1_booksim_area* num_chip1_sofar1+ chip1_area*num_chip1_sofar1*(CLUSTER1**2) + chip2_booksim_area*num_chip2_sofar1+ chip2_area*num_chip2_sofar1*(CLUSTER2**2) 
      selected_list1.append(1)
      
      num_chip1_sofar2,num_chip2_sofar2 = number_of_tile_sofar(mapping_info2)
      total_latency2 = total_latency + booksim_temp2[0] + before_conv_result2[0]
      total_energy2 = total_energy + booksim_temp2[1] + before_conv_result2[1]
      leakage2 = (cluster_booksim_leakage/(chip_width**2)*chip_width + (cluster_booksim_leakage/(chip_width**2) + chip1_booksim_leakage) * num_chip1_sofar2 + chip1_leakage*num_chip1_sofar2*(CLUSTER1**2) + (cluster_booksim_leakage/(chip_width**2)+chip2_booksim_leakage) * num_chip2_sofar2 + chip2_leakage*num_chip2_sofar2*(CLUSTER2**2) )* total_latency2 * 1000
      total_area2 = cluster_booksim_area + chip1_booksim_area* num_chip1_sofar2+ chip1_area*num_chip1_sofar2*(CLUSTER1**2) + chip2_booksim_area*num_chip2_sofar2+ chip2_area*num_chip2_sofar2*(CLUSTER2**2) 
      selected_list2.append(2)
      
      # print(before_conv_result1[1])
      # print(booksim_temp1[1])
      # print(total_energy)
      # print("num_of_col1: ",num_of_col1 , " num_of_col2: ", num_of_col2)
      # print( "chip1: ",  total_latency1, total_energy1,leakage1,total_area1, " chip2: ",total_latency2, total_energy2,leakage2,total_area2)  
      
      return iter1,level1,count1,tile_grid1,mapping_info1,conv_dense,total_latency1,total_energy1,total_area1,leakage1,selected_list1, iter2,level2,count2,tile_grid2,mapping_info2,conv_dense,total_latency2,total_energy2,total_area2,leakage2,selected_list2
    
      

def my_func(input):
    config = input[0]
    exec_set =input[1]
    config_origin = find_origin_config[f'{config}']
    iter1_tmp,level1_tmp,count1_tmp,tile_grid1_tmp,mapping_info1_tmp,conv_dense_tmp,total_latency1_tmp,total_energy1_tmp,total_area1_tmp,leakage1_tmp,selected_list1_tmp, iter2_tmp,level2_tmp,count2_tmp,tile_grid2_tmp,mapping_info2_tmp,conv_dense_tmp,total_latency2_tmp,total_energy2_tmp,total_area2_tmp,leakage2_tmp,selected_list2_tmp  = make_args(config_origin, dfg, shape1[f'{config_origin}'], shape2[f'{config_origin}'], compute_PPA1_list[f'{config}'], compute_PPA2_list[f'{config}'], chip_width_list[f'{config}'], FPS_list[f'{config}'], iter_list[f'{config}'], level_list[f'{config}'], count_list[f'{config}'], tile_grid_list[f'{config}'], mapping_info_list[f'{config}'], conv_dense_list[f'{config}'], total_latency_list[f'{config}'], total_energy_list[f'{config}'],exec_set, all_selected_list[f'{config}'])
    with lock:
      run_list.remove(f'{config}')
      selecte1 = f'_{selected_list1_tmp[-1]}'
      compute_PPA1_list[f'{config}{selecte1}'] = compute_PPA1_list[f'{config}']
      compute_PPA2_list[f'{config}{selecte1}'] = compute_PPA2_list[f'{config}']
      chip_width_list[f'{config}{selecte1}'] = chip_width_list[f'{config}']
      FPS_list[f'{config}{selecte1}'] = FPS_list[f'{config}']
      iter_list[f'{config}{selecte1}']=iter1_tmp
      level_list[f'{config}{selecte1}']=level1_tmp
      count_list[f'{config}{selecte1}']=count1_tmp
      tile_grid_list[f'{config}{selecte1}']=tile_grid1_tmp
      mapping_info_list[f'{config}{selecte1}']=mapping_info1_tmp
      conv_dense_list[f'{config}{selecte1}']=conv_dense_tmp
      total_latency_list[f'{config}{selecte1}']=total_latency1_tmp
      total_energy_list[f'{config}{selecte1}']=total_energy1_tmp
      leakage_list[f'{config}{selecte1}']=leakage1_tmp
      total_area_list[f'{config}{selecte1}']=total_area1_tmp
      all_selected_list[f'{config}{selecte1}']=selected_list1_tmp
      find_origin_config[f'{config}{selecte1}']=find_origin_config[f'{config}']
      run_list.append(f'{config}{selecte1}')

      selecte2 = f'_{selected_list2_tmp[-1]}'
      compute_PPA1_list[f'{config}{selecte2}'] = compute_PPA1_list[f'{config}']
      compute_PPA2_list[f'{config}{selecte2}'] = compute_PPA2_list[f'{config}']
      chip_width_list[f'{config}{selecte2}'] = chip_width_list[f'{config}']
      FPS_list[f'{config}{selecte2}'] = FPS_list[f'{config}']
      iter_list[f'{config}{selecte2}']=iter2_tmp
      level_list[f'{config}{selecte2}']=level2_tmp
      count_list[f'{config}{selecte2}']=count2_tmp
      tile_grid_list[f'{config}{selecte2}']=tile_grid2_tmp
      mapping_info_list[f'{config}{selecte2}']=mapping_info2_tmp
      conv_dense_list[f'{config}{selecte2}']=conv_dense_tmp
      total_latency_list[f'{config}{selecte2}']=total_latency2_tmp
      total_energy_list[f'{config}{selecte2}']=total_energy2_tmp
      leakage_list[f'{config}{selecte2}']=leakage2_tmp
      total_area_list[f'{config}{selecte2}']=total_area2_tmp
      all_selected_list[f'{config}{selecte2}']=selected_list2_tmp
      find_origin_config[f'{config}{selecte2}']=find_origin_config[f'{config}']
      run_list.append(f'{config}{selecte2}')

      del compute_PPA1_list[f'{config}']
      del compute_PPA2_list[f'{config}']
      del chip_width_list[f'{config}']
      del FPS_list[f'{config}']
      del iter_list[f'{config}']
      del level_list[f'{config}']
      del count_list[f'{config}']
      del tile_grid_list[f'{config}']
      del mapping_info_list[f'{config}']
      del conv_dense_list[f'{config}']
      del total_latency_list[f'{config}']
      del total_energy_list[f'{config}']
      del total_area_list[f'{config}']
      del all_selected_list[f'{config}'] 
      del find_origin_config[f'{config}']
      del leakage_list[f'{config}']
      

  

LATENCY={}
POWER={}
AREA={}
leakage_POWER={}
inter_connect_LATENCY={}
inter_connect_POWER={}
inter_connect_AREA={}
tile_width_meter={}
clk_frequency={}
clk_period={}
unitLatencyRep={}
unitLatencyWire={}
MAX_period = {}
minDist={}
busWidth={}
ChipArea={}
All_layer = []

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
  


for config in CONFIG:
  NUM_ROW = config[0]
  NUM_COL = config[1]
  PE = config[2]
  Tile = config[3]
  fname = f"./summary/summary_{args.model}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}.txt"
  with open(fname) as f:
    lines = f.readlines()
    for i, l in enumerate(lines):
      if l.find("readLatency is:")!= -1 and l.find("layer")!= -1:
        layername = l.split("\'")[0]
        l=l.split("\'")[1]
        latency = l.split(":")[1]
        latency = float(latency[:-3])
        LATENCY[f'{layername}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']=latency
        if layername not in All_layer:
          All_layer.append(layername)
      elif l.find("readDynamicEnergy is:")!= -1 and l.find("layer")!= -1:
        layername = l.split("\'")[0]
        l=l.split("\'")[1]
        readDynamicEnergy = l.split(":")[1]
        readDynamicEnergy = float(readDynamicEnergy[:-3])
        POWER[f'{layername}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']=readDynamicEnergy
      elif l.find("leakagePower is:")!= -1 and l.find("layer")!= -1:
        layername = l.split("\'")[0]
        l=l.split("\'")[1]
        leakagePower = l.split(":")[1]
        leakagePower = float(leakagePower[:-3])
        leakage_POWER[f'{layername}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']=leakagePower
      elif l.find("Tile Area is:")!= -1 and l.find("layer")!= -1:
        layername = l.split("\'")[0]
        l=l.split("\'")[1]
        cimarea = l.split(":")[1]
        cimarea = float(cimarea[:-5])
        AREA[f'{layername}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']=cimarea
      elif l.find("H-tree Area is")!= -1 and l.find("layer")!= -1:
        layername = l.split("\'")[0]
        l=l.split("\'")[1]
        inter_area = l.split(":")[1]
        inter_area = float(inter_area[:-5])
        inter_connect_AREA[f'{layername}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']=inter_area
      elif l.find("H-tree Latency is")!= -1 and l.find("layer")!= -1:
        layername = l.split("\'")[0]
        l=l.split("\'")[1]
        inter_latency = l.split(":")[1]
        inter_latency = float(inter_latency[:-3])
        inter_connect_LATENCY[f'{layername}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']=inter_latency
      elif l.find("H-tree Energy is")!= -1 and l.find("layer")!= -1:
        layername = l.split("\'")[0]
        l=l.split("\'")[1]
        inter_energy = l.split(":")[1]
        inter_energy = float(inter_energy[:-3])
        inter_connect_POWER[f'{layername}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']=inter_energy
      elif l.find("unitLatencyRep")!= -1:
        LatencyRep = l.split(":")[1].split("unitLatencyWire is")[0].strip()
        LatencyWire = l.split(":")[1].split("unitLatencyWire is")[1].strip()
        unitLatencyRep[f'SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']= float(LatencyRep)
        unitLatencyWire[f'SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']= float(LatencyWire)
      elif l.find("Tilewidth")!= -1:
        Tilewidth = l.split(":")[1]
        Tilewidth = float(Tilewidth[:-2])
        tile_width_meter[f'SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']= Tilewidth
      elif l.find("Chip clock period")!= -1:
        clock_period = l.split(":")[1]
        clock_period = float(clock_period[:-3])
        # if clock_period == 0:
        clock_period = (6.50252e-3)*20
        # clock_period = 0.1
        clock_freq = (1/clock_period)*1e+9
        clk_period[f'SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']= clock_period
        clk_frequency[f'SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']= clock_freq
      elif l.find("minDist")!= -1:
        Dist = l.split("minDist")[1]
        Dist = float(Dist[:-2])
        minDist[f'SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']= Dist
      elif l.find("busWidth")!= -1:
        bus = float(l.split("busWidth")[1])
        busWidth[f'SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}']= 128


x = torch.randn(args.batch_size,3,224,224)
assert args.model in ['ResNet50','EfficientB0','Mobilenet_V2'], args.model
if args.model == 'ResNet50':
    modelCF = onnx.load("resnet50.onnx")
elif args.model == 'EfficientB0':
    modelCF = onnx.load("efficientnet_b0.onnx")
elif args.model == 'Mobilenet_V2':
    modelCF = onnx.load("mobilenet_v2.onnx")
else:
    raise ValueError("Unknown model type")


graph, params = tvm.relay.frontend.from_onnx(modelCF, shape=None, dtype='float32', opset=None, freeze_params=True, convert_config=None)
# graph, params= tvm.relay.optimize(graph, target='llvm', params=params)
graph_ir = str(graph)
parse_ir = graph_ir.split('\n')
# print(graph_ir)
node={}
dfg={}


for i in parse_ir:
    # if "group" in i:
    #    continue

    name = ""
    type = ""
    input = ""
    input2 = ""
    kernel_size = ""
    stride = ""
    
    o = i.partition("/*")[0]
    j = o.partition("%")[2]
    k = j.partition("=")
    name = k[0]
    if "stride" in i:
      stride = i.partition("strides=")[2].partition(",")[0].partition("[")[2]    
    if "kernel_size" in i:
      kernel_size = i.partition("kernel_size=")[2].partition(",")[0].partition("[")[2]

    if "(" in k[2]:
        type = k[2].partition("(")[0]
        if "," in k[2]:
            if "[" in k[2]:
                input = k[2].partition("(")[2].partition(",")[0].partition("%")[2]
            else:
                input = k[2].partition("(")[2].partition(",")[0].partition("%")[2]
                input2 = k[2].partition("(")[2].partition(",")[2].partition(")")[0].partition("%")[2]
        else:
            input = k[2].partition("%")[2].partition(")")[0]
        # print(input)
    else:
        type =""
        input = k[2].partition("%")[2]

    activation = i.rpartition("/*")[2].partition("(")[2].partition(")")[0].split(",")
    if type.strip() == "nn.batch_norm":
      activation = i.rpartition("/*")[2].partition("(")[2].partition("(")[2].partition(")")[0].split(",")
    
    for h in range(len(activation)):
        activation[h] = activation[h].strip()
    if input != "":
      if "group" in i:
        node[name.strip()]=["depthwise",input.strip(),input2.strip(),activation,kernel_size,stride]
      else:
        node[name.strip()]=[type.strip(),input.strip(),input2.strip(),activation,kernel_size,stride]
last_node=None
for i in node.keys():

  if str(i) != '0':
    if node[str(i)][1] != '':
      if '.' in node[str(i)][1]:
        node[str(i)][1] =  str(int(float(node[str(i)][1])))
    if node[str(i)][2] != '':
      if '.' in node[str(i)][2]:
        node[str(i)][2] =  str(int(float(node[str(i)][2])))
for i in node.keys():
  last_node = i
  if node[str(i)][0]!='nn.dense' and node[str(i)][0]!='nn.conv2d' and node[str(i)][0]!='non-MAC':
    for j in node.keys():
      if node[str(j)][1]==str(i):
        if node[str(j)][0]=="nn.conv2d" :
          node[str(i)][0]="non-MAC"
          break
        else:
          has_two=0
          if node[str(i)][1] != '':
            node[str(j)][1]=str(int(float(node[str(i)][1])))
            has_two=1
          if node[str(i)][2] != '':
            if has_two == 1 and (args.model == "ResNet50" or args.model== "DenseNet40"):
              node[str(j)][2]=str(int(float(node[str(i)][2])))
            else:
              node[str(j)][1]=str(int(float(node[str(i)][2])))
      
      if node[str(j)][2]==str(i):
        if node[str(j)][0]=="nn.conv2d" :
          node[str(i)][0]="non-MAC"
          break
        else:
          has_two=0
          if node[str(i)][1] != '':
            node[str(j)][2]=str(int(float(node[str(i)][1])))
            has_two=1
          if node[str(i)][2] != '':
            if has_two == 1 and (args.model == "ResNet50" or args.model== "DenseNet40"):
              node[str(j)][1]=str(int(float(node[str(i)][2])))
            else:
              node[str(j)][2]=str(int(float(node[str(i)][2])))

pop_list=[]
for i in node.keys():
    if node[str(i)][0]!='nn.dense' and node[str(i)][0]!='nn.conv2d' and node[str(i)][0]!='non-MAC':
      pop_list.append(i)

for i in pop_list:
    node.pop(str(i),None)





for node_key in node.keys():
    key = node_key
    op= node[node_key][0]
    act= node[node_key][3]
    activation_size = None
    for act_ in act:
        if activation_size == None:
            activation_size = int(act_)
        else:
            activation_size = activation_size * int(act_)
    output1=''
    output2=''
    
    for tmp_key in node.keys():
        if node[tmp_key][1]==node_key:
          
          if output1=='':
              output1=tmp_key
          else:
              output2=tmp_key
        if node[tmp_key][2]==node_key:
            if output1=='':
                output1=tmp_key
            else:
                output2=tmp_key
 
    dfg[key]=[op,output1,output2,act,activation_size,1,node[node_key][4],node[node_key][5]]






path = f"/root/hetero-neurosim-search/Inference_pytorch/record_{args.model}"
if not os.path.isdir(path):
  os.makedirs(path)
  os.makedirs(f"{path}/CLUSTER")
  os.makedirs(f"{path}/SMALL")
  os.makedirs(f"{path}/BOOKSIM")
 
print(dfg)

CONFIG_pareto=[]
for layer_num in All_layer: 
  x = np.array([])
  y = np.array([])
  z = np.array([])
  labels = np.array([])
  for key in LATENCY:
    if key.find(f"{layer_num}_")!=-1:
      x=np.append(x,LATENCY.get(key,0)-inter_connect_LATENCY.get(key,0))
      y=np.append(y,(POWER.get(key,0)+leakage_POWER.get(key,0)-inter_connect_POWER.get(key,0))/(LATENCY.get(key,0)-inter_connect_LATENCY.get(key,0)))
      z=np.append(z,(AREA.get(key,0)-inter_connect_AREA.get(key,0)))
      labels=np.append(labels,key) 


  point = [list(element) for element in zip(x,y,z)]
  pareto_label={}
  for i,e in enumerate(point):
    pareto_label[(e[0]+e[1]+e[2]).tobytes()]= labels[i] 
  paretoPoints, dominatedPoints = simple_cull(point, dominates)
  for pareto in paretoPoints:
    SA_ROW_pareto = int(pareto_label[(pareto[0]+pareto[1]+pareto[2]).tobytes()].split(':')[1].split('_')[0])
    SA_COL_pareto = int(pareto_label[(pareto[0]+pareto[1]+pareto[2]).tobytes()].split(':')[2].split('_')[0])
    PE_pareto = int(pareto_label[(pareto[0]+pareto[1]+pareto[2]).tobytes()].split(':')[3].split('_')[0])
    Tile_pareto = int(pareto_label[(pareto[0]+pareto[1]+pareto[2]).tobytes()].split(':')[4])
    if [SA_ROW_pareto, SA_COL_pareto, PE_pareto, Tile_pareto] not in CONFIG_pareto:
      CONFIG_pareto.append([SA_ROW_pareto, SA_COL_pareto, PE_pareto, Tile_pareto])


print(CONFIG_pareto)
print(len(CONFIG_pareto))
combine_CONFIG = list(combinations(CONFIG_pareto,2))
random.shuffle(combine_CONFIG)

for j, config in enumerate(combine_CONFIG):
  if config[0][0] * config[0][3] > config[1][0] * config[1][3]:
      NUM1_ROW = config[0][0]
      NUM1_COL = config[0][1]
      PE1 = config[0][2]
      Tile1 = config[0][3]
      NUM2_ROW = config[1][0]
      NUM2_COL = config[1][1]
      PE2 = config[1][2]
      Tile2 = config[1][3]
  else:
      NUM1_ROW = config[1][0]
      NUM1_COL = config[1][1]
      PE1 = config[1][2]
      Tile1 = config[1][3]
      NUM2_ROW = config[0][0]
      NUM2_COL = config[0][1]
      PE2 = config[0][2]
      Tile2 = config[0][3]
  combine_CONFIG[j] = [[NUM1_ROW,NUM1_COL,PE1,Tile1],[NUM2_ROW,NUM2_COL,PE2,Tile2]]
  

execute_set=[]
tmp_set=[]
print(dfg.keys())
for dfg_node in dfg.keys():
  if dfg[dfg_node][0] == 'nn.conv2d' or dfg[dfg_node][0] == 'nn.dense':
    if len(tmp_set) > 0:
      execute_set.append(tmp_set)
      tmp_set=[dfg_node]
    else:
      tmp_set=[dfg_node]
  else:
    tmp_set.append(dfg_node)
    execute_set.append(tmp_set)
    tmp_set=[]
execute_set.append(tmp_set)
  
print(execute_set)
# exit()
shape1=Manager().dict()
shape2=Manager().dict()
compute_PPA1_list=Manager().dict()
compute_PPA2_list=Manager().dict()
chip_width_list=Manager().dict()
FPS_list=Manager().dict()
# change value dependent choosing tile
iter_list=Manager().dict()
level_list=Manager().dict()
count_list=Manager().dict()
tile_grid_list=Manager().dict()
mapping_info_list=Manager().dict()
conv_dense_list=Manager().dict()
total_latency_list=Manager().dict()
total_energy_list=Manager().dict()
total_area_list=Manager().dict()
all_selected_list=Manager().dict()
leakage_list=Manager().dict()
run_list = Manager().list()
find_origin_config = Manager().dict()


depth = -1
if not os.path.isdir(f"{path}/CLUSTER/{depth}"):
  os.makedirs(f"{path}/CLUSTER/{depth}")
  os.makedirs(f"{path}/SMALL/{depth}")
  os.makedirs(f"{path}/BOOKSIM/{depth}")
for config in combine_CONFIG:
  shape1[f'{config}'],shape2[f'{config}'],compute_PPA1_list[f'{config}'],compute_PPA2_list[f'{config}'],chip_width_list[f'{config}'],FPS_list[f'{config}'],iter_list[f'{config}'],level_list[f'{config}'],count_list[f'{config}'],tile_grid_list[f'{config}'],mapping_info_list[f'{config}'],conv_dense_list[f'{config}'],total_latency_list[f'{config}'],total_energy_list[f'{config}'],total_area_list[f'{config}'],leakage_list[f'{config}'],all_selected_list[f'{config}'] = initialization(config)
  run_list.append(f'{config}')
  find_origin_config[f'{config}'] = config

results_dict = Manager().dict()
lock = Manager().Lock() 

NCPU=multiprocessing.cpu_count()
depth = 0
for exec_set in execute_set:
  # print("combine_CONFIG",combine_CONFIG)
  print(exec_set)
  print(f"------------------{depth}-----------------")
  time.sleep(1)
  if not os.path.isdir(f"{path}/CLUSTER/{depth}"):
    os.makedirs(f"{path}/CLUSTER/{depth}")
    os.makedirs(f"{path}/SMALL/{depth}")
    os.makedirs(f"{path}/BOOKSIM/{depth}")
    if os.path.isdir(f"{path}/CLUSTER/{depth-2}"):
      shutil.rmtree(f"{path}/CLUSTER/{depth-2}")
      shutil.rmtree(f"{path}/SMALL/{depth-2}")
      shutil.rmtree(f"{path}/BOOKSIM/{depth-2}")
  if not os.path.isdir(f"{path}/Prepared_data/{depth}"):
    os.makedirs(f"{path}/Prepared_data/{depth}")
  exec_set_list_repeated = [[combine_CONFIG[i],exec_set] for i in range(len(combine_CONFIG))] 
  with Pool(processes=NCPU-3) as pool:
    pool.map(my_func, exec_set_list_repeated)
  print(f"end_pool")
  alive_combine_CONFIG = []
  if depth > 0 :
    x = np.array([])
    y = np.array([])
    z = np.array([])
    labels = np.array([])
    for r in run_list:
      x=np.append(x,total_latency_list[r])
      y=np.append(y,(total_energy_list[r]+leakage_list[r])/total_latency_list[r])
      z=np.append(z,total_area_list[r])
      labels=np.append(labels,r) 
    point = [list(element) for element in zip(x,y,z)]
    pareto_label={}
    
    for i,e in enumerate(point):
      if str(e[0])+str(e[1])+str(e[2]) in pareto_label.keys():
        pareto_label[str(e[0])+str(e[1])+str(e[2])].append(labels[i])
      else:
        pareto_label[str(e[0])+str(e[1])+str(e[2])]= [labels[i]]
    
    paretoPoints, dominatedPoints = simple_cull(point, dominates)
    print("len_pareto",paretoPoints)
    if len(paretoPoints) > args.beam_size_m:
      paretoPoints_list=[]
      for pareto in paretoPoints:
        tmp_pareto =[pareto[0],pareto[1],pareto[2]]
        paretoPoints_list.append(tmp_pareto)
        

      w = [args.latency,args.power,args.area]
      sign = np.array([False,False,False])
      t = Topsis(paretoPoints_list, w, sign)
      t.calc()
      beam_list=[]
      beam_list_score={}
      overlap_count = {} 
      print(len(t.rank_to_best_similarity()))
      for beam in range(len(t.rank_to_best_similarity())):
        print("beam",overlap_count.values())
        if sum(overlap_count.values()) < args.beam_size_m:
          pal = pareto_label[str(paretoPoints_list[t.rank_to_best_similarity()[beam]-1][0])+str(paretoPoints_list[t.rank_to_best_similarity()[beam]-1][1])+str(paretoPoints_list[t.rank_to_best_similarity()[beam]-1][2])][0]
          print("pal.split()[0]",pal.split("_")[0])
          if pal.split("_")[0] in overlap_count:
            if overlap_count[pal.split("_")[0]] < args.beam_size_n:
              beam_list.append(paretoPoints_list[t.rank_to_best_similarity()[beam]-1])
              beam_list_score[pal]=t.best_similarity[beam]
              overlap_count[pal.split("_")[0]] +=1
              print(overlap_count[pal.split("_")[0]])
            else:
              continue
          else:
            beam_list.append(paretoPoints_list[t.rank_to_best_similarity()[beam]-1])
            beam_list_score[pal]=t.best_similarity[beam]
            overlap_count[pal.split("_")[0]] =1
        else:
          break
        
      paretoPoints = beam_list
    
    
    run_list = Manager().list()
    for pareto in paretoPoints:
      for ad in pareto_label[str(pareto[0])+str(pareto[1])+str(pareto[2])]:
        alive_combine_CONFIG.append(ad)
        run_list.append(ad)
    # for dominate in dominatedPoints:
    #   for rem in pareto_label[str(dominate[0])+str(dominate[1])+str(dominate[2])]:
    #     run_list.remove(rem)
  
  
  else:
    for r in run_list:
      alive_combine_CONFIG.append(r)
  
  combine_CONFIG = alive_combine_CONFIG
  depth+=1

if not os.path.isdir(f"/root/hetero-neurosim-search/Inference_pytorch/search_result/{args.model}_hetero"):
  os.makedirs(f"/root/hetero-neurosim-search/Inference_pytorch/search_result/{args.model}_hetero")
  
final_latency_f = open(f"./search_result/{args.model}_hetero/final_result_{args.beam_size_m}_{args.beam_size_n}_{args.distribute}_[{args.latency},{args.power},{args.area}].txt",'w', newline='')
final_latency_wr = csv.writer(final_latency_f)
for final in combine_CONFIG:
  final_latency_wr.writerow([final,total_latency_list[final], (total_energy_list[final]+leakage_list[final])/total_latency_list[final],total_area_list[final]])
final_latency_f.close()

final_log_f = open(f"./search_result/{args.model}_hetero/final_log_{args.beam_size_m}_{args.beam_size_n}_{args.distribute}_[{args.latency},{args.power},{args.area}].txt",'w', newline='')
final_log_wr = csv.writer(final_log_f)
for final_ in combine_CONFIG:
  num_chip1_sofar1,num_chip2_sofar1 = number_of_tile_sofar(mapping_info_list[final_])
  final_log_wr.writerow([final_,mapping_info_list[final_],num_chip1_sofar1,num_chip2_sofar1])
final_log_f.close()
  
  
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")



