import argparse
import os
from subprocess import Popen

parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--model', default='ResNet50', help='VGG16|ResNet50|NasNetA|LFFD')
parser.add_argument('--beam_size_m',type=int ,default=500,help='beam_size_m')
parser.add_argument('--beam_size_n',type=int ,default=3,help='beam_size_n')
parser.add_argument('--weight_latency',type=int ,default=1,help='weight_latency_with_pareto')
parser.add_argument('--weight_power',type=int ,default=1,help='weight_power_with_pareto')
parser.add_argument('--weight_area',type=int ,default=1,help='weight_area_with_pareto')
args = parser.parse_args()




os.system(f"pueue add python profile_booksim_hetero.py --model={args.model} --beam_size_m={args.beam_size_m} --beam_size_n={args.beam_size_n} --latency={args.weight_latency} --power={args.weight_power} --area={args.weight_area}")

    


