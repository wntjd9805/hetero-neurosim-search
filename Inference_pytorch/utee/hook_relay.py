#from modules.quantize import quantize, quantize_grad, QConv2d, QLinear, RangeBN
import os
import torch.nn as nn
import shutil
from modules.quantization_cpu_np_infer import QConv2d,QLinear
from modules.floatrange_cpu_np_infer import FConv2d, FLinear
import numpy as np
import torch
from utee import wage_quantizer
from utee import float_quantizer
from parse import *
import re

def Neural_Sim(input_shape, weight_shape, pad, name, iter, wl_input, wl_weight, strides): 
    global model_n, FP
    input_parse = re.findall(r'\d+', str(input_shape))
    weight_parse = re.findall(r'\d+', str(weight_shape))
    if name == "Conv_":
        input = torch.randn(int(input_parse[0]),int(input_parse[1]),int(input_parse[2]),int(input_parse[3]))
        weight = torch.randn(int(weight_parse[0]),int(weight_parse[1]),int(weight_parse[2]),int(weight_parse[3]))
    elif name == "FC_":
        input = torch.randn(int(input_parse[0]),int(input_parse[1]))
        weight = torch.randn(int(weight_parse[0]),int(weight_parse[1]))

    print("quantize layer ", name, iter)
    input_file_name =  './layer_record_' + str(model_n) + '/input' + str(name) + str(iter)+ '.csv'
    weight_file_name =  './layer_record_' + str(model_n) + '/weight' + str(name) + str(iter) + '.csv'
    f = open('./layer_record_' + str(model_n) + '/trace_command.sh', "a")
    f.write(weight_file_name+' '+input_file_name+' ')
    if FP:
        weight_q = float_quantizer.float_range_quantize(weight,wl_weight)
    else:
        weight_q = wage_quantizer.Q(weight,wl_weight)
    write_matrix_weight( weight_q.cpu().data.numpy(),weight_file_name)
    if len(weight.shape) > 2:
        k=weight.shape[-1]
        pad_parse = re.findall(r'\d+', str(pad))
        padding = (int(pad_parse[0]),int(pad_parse[1]))
        stride_parse = re.findall(r'\d+', str(strides))
        stride=(int(stride_parse[0]),int(stride_parse[1]))
        write_matrix_activation_conv(stretch_input(input.cpu().data.numpy(),k,padding,stride),None,wl_input,input_file_name)
    else:
        write_matrix_activation_fc(input.cpu().data.numpy(),None ,wl_input, input_file_name)

def write_matrix_weight(input_matrix,filename):
    cout = input_matrix.shape[0]
    weight_matrix = input_matrix.reshape(cout,-1).transpose()
    np.savetxt(filename, weight_matrix, delimiter=",",fmt='%10.5f')


def write_matrix_activation_conv(input_matrix,fill_dimension,length,filename):
    filled_matrix_b = np.zeros([input_matrix.shape[2],input_matrix.shape[1]*length],dtype=np.str)
    filled_matrix_bin,scale = dec2bin(input_matrix[0,:],length)
    for i,b in enumerate(filled_matrix_bin):
        filled_matrix_b[:,i::length] =  b.transpose()
    np.savetxt(filename, filled_matrix_b, delimiter=",",fmt='%s')


def write_matrix_activation_fc(input_matrix,fill_dimension,length,filename):

    filled_matrix_b = np.zeros([input_matrix.shape[1],length],dtype=np.str)
    filled_matrix_bin,scale = dec2bin(input_matrix[0,:],length)
    for i,b in enumerate(filled_matrix_bin):
        filled_matrix_b[:,i] =  b
    np.savetxt(filename, filled_matrix_b, delimiter=",",fmt='%s')


def stretch_input(input_matrix,window_size = 5,padding=(0,0),stride=(1,1)):
    input_shape = input_matrix.shape
    print(input_shape)
    item_num = ((input_shape[2] + 2*padding[0] - window_size) / stride[0] + 1) * ((input_shape[3] + 2*padding[1] - window_size) / stride[1] + 1)
    output_matrix = np.zeros((input_shape[0],int(item_num),input_shape[1]*window_size*window_size))
    iter = 0
    
    print(window_size)
    print(padding)
    print(stride)
    for i in range( int((input_shape[2] - window_size) / stride[0] + 1 )):
        for j in range( int((input_shape[3] - window_size) / stride[1] + 1)):
            for b in range(input_shape[0]):
                output_matrix[b,iter,:] = input_matrix[b, :, i:i+window_size,j: j+window_size].reshape(input_shape[1]*window_size*window_size)
            iter += 1

    return output_matrix


def dec2bin(x,n):
    y = x.copy()
    out = []
    scale_list = []
    delta = 1.0/(2**(n-1))
    x_int = x/delta

    base = 2**(n-1)

    y[x_int>=0] = 0
    y[x_int< 0] = 1
    rest = x_int + base*y
    out.append(y.copy())
    scale_list.append(-base*delta)
    for i in range(n-1):
        base = base/2
        y[rest>=base] = 1
        y[rest<base]  = 0
        rest = rest - base * y
        out.append(y.copy())
        scale_list.append(base * delta)

    return out,scale_list

def bin2dec(x,n):
    bit = x.pop(0)
    base = 2**(n-1)
    delta = 1.0/(2**(n-1))
    y = -bit*base
    base = base/2
    for bit in x:
        y = y+base*bit
        base= base/2
    out = y*delta
    return out

def remove_hook_list(hook_handle_list):
    for handle in hook_handle_list:
        handle.remove()

def hardware_evaluation(graph,wl_weight,wl_activation,model_name,mode,input_shape,weight_shape,name,iter,pad,stride): 
    global model_n, FP
    model_n = model_name
    FP = 1 if mode=='FP' else 0
    
    hook_handle_list = []
    

    Neural_Sim(input_shape, weight_shape, pad, name, iter, wl_activation, wl_weight, stride)
    # for i, layer in enumerate(model.modules()):
    #     if isinstance(layer, (FConv2d, QConv2d, nn.Conv2d)) or isinstance(layer, (FLinear, QLinear, nn.Linear)):
    #         hook_handle_list.append(Neural_Sim(self))
    return hook_handle_list
