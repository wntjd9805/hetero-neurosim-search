

import argparse
import csv
import os
import shutil
parser = argparse.ArgumentParser()
parser.add_argument("--model",  required=True)
args = parser.parse_args()

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



if not os.path.isdir(f"./neurosim_result"):
  os.makedirs(f"./neurosim_result")
if not os.path.isdir(f"./summary"):
  os.makedirs(f"./summary")
if not os.path.isdir(f"./shape"):
  os.makedirs(f"./shape")
for i in CONFIG:
  NUM_ROW = i[0]
  NUM_COL = i[1]
  PE = i[2]
  Tile = i[3]

  fname = f"{args.model}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}.txt"
  readLatency=[]
  power=[]
  summary=[]
  with open(fname) as f:
    lines = f.readlines()
    t = 0; 
    for i, l in enumerate(lines):
      if l.find("readLatency is:")!= -1 and l.find("layer")!= -1:
        l =l.strip('\n')
        readLatency.append(l)
      if l.find("readDynamicEnergy is:")!= -1 and l.find("layer")!= -1:
        l =l.strip('\n')
        readLatency.append(l)
      if l.find("leakageEnergy is:")!= -1 and l.find("layer")!= -1:
        l =l.strip('\n')
        readLatency.append(l)
      if l.find("leakagePower is:")!= -1 and l.find("layer")!= -1:
        l =l.strip('\n')
        readLatency.append(l)
      if l.find("Tile Area is:")!= -1 and l.find("layer")!= -1:
        l =l.strip('\n')
        readLatency.append(l)
      if l.find("H-tree Area is:")!= -1 and l.find("layer")!= -1:
        l =l.strip('\n')
        readLatency.append(l)
      if l.find("H-tree Latency is")!= -1:
        l =l.strip('\n')
        readLatency.append(l)
      if l.find("H-tree Energy is")!= -1:
        l =l.strip('\n')
        readLatency.append(l)
      if l.find("unitLatencyRep is")!= -1 and t == 0:
        l =l.strip('\n')
        readLatency.append(l)
        t = 1  
      if l.find("Summary") != -1:
        for k in range(len(lines)-i):
          l =lines[i+k].strip('\n')
          summary.append(l)    
  f = open(f'./summary/summary_{args.model}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}.txt','w', newline='')
  wr = csv.writer(f)
  for c in readLatency:
    wr.writerow([c])
  for c in summary:
    wr.writerow([c])
  shutil.move(f"shape_{args.model}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}",f"./shape/shape_{args.model}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}")
  shutil.move(f"{args.model}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}.txt",f"./neurosim_result/{args.model}_SA_row:{NUM_ROW}_SA_col:{NUM_COL}_PE:{PE}_TL:{Tile}.txt")


