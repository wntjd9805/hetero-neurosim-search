import argparse
import os
import time
from utee import misc
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utee import make_path
from utee import wage_util
from models import dataset
import torchvision.models as models
from utee import hook_relay
#from IPython import embed
from datetime import datetime
from subprocess import call

import tvm 
import tvm.relay as relay
import onnx
from parse import *
import re
import csv

   

parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--dataset', default='imagenet', help='imagenet')
parser.add_argument('--model', default='VGG16', help='VGG16|ResNet50|NasNetA|LFFD')
parser.add_argument('--mode', default='WAGE', help='WAGE|FP')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 10)')
parser.add_argument('--grad_scale', type=float, default=8, help='learning rate for wage delta calculation')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100,  help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1,  help='how many epochs to wait before another test')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 1e-3)')
parser.add_argument('--decreasing_lr', default='140,180', help='decreasing strategy')
parser.add_argument('--wl_weight', default=8)
parser.add_argument('--wl_grad', default=8)
parser.add_argument('--wl_activate', default=8)
parser.add_argument('--wl_error', default=8)
parser.add_argument('--tvm_optimization', default=0)
# Hardware Properties
# if do not consider hardware effects, set inference=0
parser.add_argument('--inference', default=0, help='run hardware inference simulation')
parser.add_argument('--subArray', default=128, help='size of subArray (e.g. 128*128)')
parser.add_argument('--ADCprecision', default=5, help='ADC precision (e.g. 5-bit)')
parser.add_argument('--cellBit', default=4, help='cell precision (e.g. 4-bit/cell)')
parser.add_argument('--onoffratio', default=10, help='device on/off ratio (e.g. Gmax/Gmin = 3)')
# if do not run the device retention / conductance variation effects, set vari=0, v=0
parser.add_argument('--vari', default=0, help='conductance variation (e.g. 0.1 standard deviation to generate random variation)')
parser.add_argument('--t', default=0, help='retention time')
parser.add_argument('--v', default=0, help='drift coefficient')
parser.add_argument('--detect', default=0, help='if 1, fixed-direction drift, if 0, random drift')
parser.add_argument('--target', default=0, help='drift target for fixed-direction drift')
current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

args = parser.parse_args()

args.logdir = os.path.join(os.path.dirname(__file__), args.logdir)
args = make_path.makepath(args,['log_interval','test_interval','logdir','epochs','gpu','ngpu','debug'])

misc.logger.init(args.logdir, 'test_log' + current_time)
logger = misc.logger.info

misc.ensure_dir(args.logdir)
logger("=================FLAGS==================")
for k, v in args.__dict__.items():
	logger('{}: {}'.format(k, v))


# seed
args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

# data loader and model
# assert args.dataset in ['imagenet'], args.dataset
# if args.dataset == 'imagenet':
#     train_loader, test_loader = dataset.get_imagenet(batch_size=args.batch_size, num_workers=1)
# else:
#     raise ValueError("Unknown dataset type")


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



# EP_list = ['CPUExecutionProvider']
# sess = rt.InferenceSession("resnet50_transformed.onnx", providers=EP_list)
# output_name = sess.get_outputs()[0].name
# input_name = sess.get_inputs()[0].name

# best_acc, old_file = 0, None
# t_begin = time.time()
# # ready to go

# test_loss = 0
# correct = 0
# trained_with_quantization = True

# criterion = torch.nn.CrossEntropyLoss()
# criterion = wage_util.SSE()

# for data, target in test_loader:
# for i, (data, target) in enumerate(test_loader):
#     ort_inputs = {ort_session.get_inputs()[0].name: data}
#     if i==0:
#         hook_handle_list = hook.hardware_evaluation(modelCF,args.wl_weight,args.wl_activate,args.model,args.mode)
#     indx_target = target.clone()
#     if args.cuda:
#         data, target = data.cuda(), target.cuda()
#     with torch.no_grad():
#         data, target = Variable(data), Variable(target)
#         ort_outs = ort_session.run(None, ort_inputs)
#         test_loss_i = criterion(ort_outs, target)
#         test_loss += test_loss_i.data
#         pred = output.data.max(1)[1]  # get the index of the max log-probability
#         correct += pred.cpu().eq(indx_target).sum()
#     if i==0:
#         hook.remove_hook_list(hook_handle_list)

# test_loss = test_loss / len(test_loader)  # average over number of mini-batch
# acc = 100. * correct / len(test_loader.dataset)

# accuracy = acc.cpu().data.numpy()
graph, params = tvm.relay.frontend.from_onnx(modelCF, shape=None, dtype='float32', opset=None, freeze_params=True, convert_config=None)
print(type(args.tvm_optimization))
print(graph)
print(params)
if args.tvm_optimization == '1':
    graph, params= tvm.relay.optimize(graph, target='llvm', params=params)
    args.model = str(args.model)+'_tvm_optimize'
    

graph_ir = str(graph)
parse_ir = graph_ir.split('\n')
output_shape = {"input": (args.batch_size,3,224,224),"data": (args.batch_size,3,224,224)}
followed_pool = []
iter_conv = 0
iter_fc = 0

if not os.path.exists('layer_record_'+str(args.model)):
    os.makedirs('layer_record_'+str(args.model))
if os.path.exists('layer_record_'+str(args.model)+'/trace_command.sh'):
    os.remove('layer_record_'+str(args.model)+'/trace_command.sh')
f = open('layer_record_'+str(args.model)+'/trace_command.sh', "w")
f.write('NeuroSIM/main ./NeuroSIM/NetWork_'+str(args.model)+'.csv '+str(args.wl_weight)+' '+str(args.wl_activate)+' ')
f.close()


for i in parse_ir:
    print(i)
    if "nn.conv2d" in i:
        i = i.strip()
        if "strides" in i:
            result = parse("%{output} = nn.conv2d(%{input}, meta[relay.Constant][{n}] /* ty=Tensor[{weight_shape}, float32] {span0} */, strides={stride}, padding={padding}, channels={channel}, kernel_size={kernel_size}) /* ty=Tensor[{output_shape}, float32] {span1} */;",i)
            stride = result["stride"]
            stride_parse = re.findall(r'\d+', str(stride))
        else:
            result = parse("%{output} = nn.conv2d(%{input}, meta[relay.Constant][{n}] /* ty=Tensor[{weight_shape}, float32] {span0} */, padding={padding}, channels={channel}, kernel_size={kernel_size}) /* ty=Tensor[{output_shape}, float32] {span1} */;",i)
            stride = "[1, 1]"
            stride_parse = [1,1]
        output_shape[result["output"]] = result["output_shape"]
        input_ = output_shape[result["input"]]
        weight = result["weight_shape"]
        pad = result["padding"]
        name = "Conv_"
        input_parse = re.findall(r'\d+', str(input_))
        weight_parse = re.findall(r'\d+', str(weight))
        if int(weight_parse[1]) != 1:
            hook_relay.hardware_evaluation(graph,args.wl_weight,args.wl_activate,args.model,args.mode,input_,weight,name,iter_conv,pad,stride)
            followed_pool.append([int(input_parse[2]),int(input_parse[3]),int(input_parse[1]),int(weight_parse[2]),int(weight_parse[3]),int(weight_parse[0]),0,int(stride_parse[0]),0])
            iter_conv = iter_conv + 1



        
    elif "nn.dense" in i:
        i = i.strip()
        result = parse("%{output} = nn.dense(%{input}, meta[relay.Constant][{n}] /* ty=Tensor[{weight_shape}, float32] {span0} */, units={weight_bias}) /* ty=Tensor[{output_shape}, float32] {span1} */;",i)
        output_shape[result["output"]] = result["output_shape"]
        input_ = output_shape[result["input"]]
        weight = result["weight_shape"]
        name = "FC_"
        hook_relay.hardware_evaluation(graph,args.wl_weight,args.wl_activate,args.model,args.mode,input_,weight,name,iter_fc,0,1)
        
        input_parse = re.findall(r'\d+', str(input_))
        weight_parse = re.findall(r'\d+', str(weight))
        followed_pool.append([1,1,int(input_parse[1]),1,1,int(weight_parse[0]),0,1])
        iter_fc = iter_fc + 1
    elif "pool" in i:
        followed_pool[-1][-3] = 1
        i = i.strip()
        result = parse("%{output} = {ww}/* ty=Tensor[{output_shape}, float32] {span0} */;",i)
        if result is not None:
            output_shape[result["output"]] = result["output_shape"]

    elif "nn.batch_norm" in i:
        i = i.strip()
        result = parse("%{output} = nn.batch_norm(%{input}, {ww} /* ty=(Tensor[{output_shape}, float32] {eee};",i)
        if result is not None:
            output_shape[result["output"]] = result["output_shape"]
    elif "split" in i:
        i = i.strip()
        result = parse("%{output} = split(%{ww}) /* ty=(Tensor[({qq}), float32], Tensor[{output_shape}, float32] {eee};",i)
        if result is not None:
            output_shape[result["output"]] = result["output_shape"]

    elif "nn.bias_add" in i:
        i = i.strip()
        result = parse("%{output} = {ww}/* ty=Tensor[{ll}, float32] {span0} */) /* ty=Tensor[{output_shape}, float32] {span1};",i)
        output_shape[result["output"]] = result["output_shape"]
        if result is not None:
            output_shape[result["output"]] = result["output_shape"]

    else:
        i = i.strip()
        result = parse("%{output} = {ww}, ty=Tensor[{output_shape}, float32] {span0} */;",i)
        result = parse("%{output} = {ww}/* ty=Tensor[{output_shape}, float32] {span0} */;",i)
        if result is not None:
            output_shape[result["output"]] = result["output_shape"]
        else:
            result = parse("%{output} = %{input}.{w};",i)
            if result is not None:
                output_shape[result["output"]] = output_shape[result["input"]]
                


with open('NeuroSIM/NetWork_'+str(args.model)+'.csv', 'w') as file:
    writer = csv.writer(file) 
    for i in followed_pool:
        writer.writerow(i)



if args.inference:
    print(" --- Hardware Properties --- ")
    print("subArray size: ")
    print(args.subArray)
    print("ADC precision: ")
    print(args.ADCprecision)
    print("cell precision: ")
    print(args.cellBit)
    print("on/off ratio: ")
    print(args.onoffratio)
    print("variation: ")
    print(args.vari)

# logger('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
# 	test_loss, correct, len(test_loader.dataset), acc))

# call(["/bin/bash", './layer_record_'+str(args.model)+'/trace_command.sh'])














