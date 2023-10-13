#!/usr/bin/python3

from subprocess import Popen
import time
import multiprocessing
import argparse

NCPU=multiprocessing.cpu_count()


parser = argparse.ArgumentParser()
parser.add_argument("--model",  required=True)
args = parser.parse_args()

def make_args(config):
  SA_ROW ,SA_COL, PE, Tile = config
  return f'bash ./layer_record_{args.model}/trace_command.sh {args.model} {SA_ROW} {SA_COL} {PE} {Tile} | tee {args.model}_SA_row:{SA_ROW}_SA_col:{SA_COL}_PE:{PE}_TL:{Tile}.txt'

def simulate(configs):
  edited_lines = []
  with open(f"./layer_record_{args.model}/trace_command.sh") as f:
    lines = f.readlines()
    if len(lines) == 1:
      for line in lines:
        edited_lines.append("MODEL=$1\n")
        edited_lines.append("SA_ROW=$2\n")
        edited_lines.append("SA_COL=$3\n")
        edited_lines.append("PE=$4\n")
        edited_lines.append("Tile=$5\n")
        update = line + " $MODEL $SA_ROW $SA_COL $PE $Tile"
        edited_lines.append(update)        
      with open(f"./layer_record_{args.model}/trace_command.sh", 'w') as f:
        f.writelines(edited_lines)

  procs = []
  n = 0
  r = 0
  for i in range(min(NCPU, len(configs))):
    procs.append(Popen(make_args(configs[i]), shell=True, start_new_session=True))
    r += 1
    time.sleep(1)

  while True:
    if n >= len(configs):
      break

    for i, p in enumerate(procs):
      if p is None:
        continue

      if p.poll() is not None:
        if r < len(configs):
          procs[i] = Popen(make_args(configs[r]), shell=True, start_new_session=True)
          r += 1
        else:
          procs[i] = None

        n += 1
        time.sleep(1)

    time.sleep(1)


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



simulate(CONFIG)