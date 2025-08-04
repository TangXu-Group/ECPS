import os
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import more_itertools as mi
import random
import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
import warnings
import sys
import math
import concurrent.futures
import argparse

warnings.filterwarnings("ignore")

from model import UNet
from tqdm import tqdm, trange
from tqdm import tqdm as tqdmm
from torch.utils.data import DataLoader
from torch.optim import Adam
from IPython.display import HTML, display,clear_output
from collections import Counter
from tool.utils import *
import tool.dataset as d
from tool.lr_scheduler import OneCycle
from itertools import cycle


def parse_args():
    parse = argparse.ArgumentParser(description='Semi-supervised change detection')  # 2、创建参数对象
    parse.add_argument('-d','--Dataset_name', type=str, required=True, help='Name of dataset')  # 3、往参数对象添加参数
    parse.add_argument('-t','--train_ratio', type=str, required=True, help='0.x')
    parse.add_argument('-s','--std_model_num', type=int, required=True, help='>=1')
    parse.add_argument('--gpus', default='0,1', type=str, help='using gpus')
    parse.add_argument('--epoch_num', default=99, type=int, help='')
    parse.add_argument('--batch_size_train', default=16, type=int, help='')
    parse.add_argument('--batch_size_test', default=32, type=int, help='')
    parse.add_argument('--num_workers', default=20, type=int, help='')
    parse.add_argument('--pin_memory', default=True, type=bool, help='')
    parse.add_argument('--p_r', default=0.35, type=float, help='')
    parse.add_argument('--pretrained_stu_paths', default=None, type=str, help='')
    parse.add_argument('--pretrained_tea_path', default=None, type=str, help='')
    args = parse.parse_args()  # 4、解析参数对象获得解析对象
    return args



def parameter_list(std_models):
    stu_paras=[]
    for model in std_models:
        stu_para = iter(model.parameters())
        stu_paras.append(stu_para)
    nn_parameter_list = []
    for i in range(len(l_module_name)):
        parameter = []
        for std_ in range(std_model_num):
            parameter.append(next(stu_paras[std_]))
        nn_parameter_list.append(nn.ParameterList(parameter))
    return nn_parameter_list

    
# balance recall and percision
def f1(x):
    return 1-x
def line1(x,a=0.5):
    y = ((x-0)*(f1(a)-0))/(a-0)+0
    return y
def line2(x,a=0.5):
    y = ((x-1)*(f1(a)-1))/(a-1)+1
    return y

#multi threads
def infer_model(model, input1, input2):
    with torch.set_grad_enabled(True):
        output = model(input1, input2)
    return output




if __name__=='__main__':
    args = parse_args()
    Dataset_name = args.Dataset_name
    gpus = args.gpus
    train_ratio = args.train_ratio
    epoch_num = args.epoch_num
    std_model_num = args.std_model_num
    w_batch_size = o_batch_size = args.batch_size_train
    test_batch_size = val_batch_size = args.batch_size_test
    num_workers = args.num_workers
    pin_memory = args.pin_memory
    p_r = args.p_r
    log_path_root = 'Run_logging_'+Dataset_name+'_'+train_ratio+'/'
    Log_path = log_path_root+Dataset_name+'_'+str(std_model_num)+'_students/'

    # pretrained_stu_paths = [Log_path+'/Best_stu_model'+str(i+1)+'.pth' for i in range(std_model_num)]
    # pretrained_tea_path = Log_path+'/Best_tea_model.pth'
    pretrained_stu_paths = None
    pretrained_tea_path = None

    def p_yr_fun(x, a=p_r):
        '''
        a: degree of paying attention to precision
        '''
        y = (x<a)*line1(x,a)+(x>=a)*line2(x,a)
        return y

    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    
    if not os.path.exists(log_path_root):
        os.mkdir(log_path_root)
    if not os.path.exists(Log_path):
        os.mkdir(Log_path)
    
    if Dataset_name == 'CDD':
        DATA_PATH = '/home/yang/yyq/Dataset/CDD'
    if Dataset_name == 'SYSU':
        DATA_PATH = '/home/yang/yyq/Dataset/SYSU-CD/'
    if Dataset_name == 'LEVIR':
        DATA_PATH = '/home/yang/yyq/Dataset/LEVIR-CD-crop/'
    
    TRAIN_DATA_PATH = os.path.join(DATA_PATH)
    TRAIN_LABEL_PATH = os.path.join(DATA_PATH)
    TRAIN_w_TXT_PATH = os.path.join('super_txt/'+Dataset_name+train_ratio+'/wlabel.txt')
    TRAIN_o_TXT_PATH = os.path.join('super_txt/'+Dataset_name+train_ratio+'/olabel.txt')
    VAL_DATA_PATH = os.path.join(DATA_PATH)
    VAL_LABEL_PATH = os.path.join(DATA_PATH)
    VAL_TXT_PATH = os.path.join(VAL_DATA_PATH,'val.txt')
    TEST_DATA_PATH = os.path.join(DATA_PATH)
    TEST_LABEL_PATH = os.path.join(DATA_PATH)
    TEST_TXT_PATH = os.path.join(TEST_DATA_PATH, 'test.txt')


    '''Loading dataset'''
    wtrain_data = d.Dataset(TRAIN_DATA_PATH, TRAIN_LABEL_PATH,
                                TRAIN_w_TXT_PATH,'train',transform=True)
    otrain_data = d.Dataset(TRAIN_DATA_PATH, TRAIN_LABEL_PATH,
                                TRAIN_o_TXT_PATH,'train',transform=True)
    wtrain_loader = DataLoader(wtrain_data, batch_size=w_batch_size,
                                 shuffle= True, num_workers= num_workers, pin_memory= pin_memory)
    otrain_loader = DataLoader(otrain_data, batch_size=o_batch_size,
                                 shuffle= True, num_workers= num_workers, pin_memory= pin_memory)
    test_data = d.Dataset(TEST_DATA_PATH, TEST_LABEL_PATH,
                            TEST_TXT_PATH,'test', transform=False)
    test_loader = DataLoader(test_data, batch_size=test_batch_size,
                                shuffle= False, num_workers= num_workers, pin_memory= pin_memory)
    val_data = d.Dataset(VAL_DATA_PATH, VAL_LABEL_PATH,
                            VAL_TXT_PATH,'val', transform=False)
    val_loader = DataLoader(val_data, batch_size=val_batch_size,
                                shuffle= False, num_workers= num_workers, pin_memory= pin_memory)


    CE = nn.CrossEntropyLoss()
    BCE = nn.BCELoss()
    MSE = nn.MSELoss()

    
    opt_num_epoch = len(otrain_loader)
    '''Student models'''
    std_models = [nn.DataParallel(UNet(3,2)).cuda() for i in range(std_model_num)] #Construct student models
    l_module_name = [i[0][7:] for i in std_models[0].named_parameters()]  #Obtain model parameter names
    nn_parameter_list = parameter_list(std_models) #Obtain their parameter list
    parameters = parameters = [param for model in std_models for param in list(model.parameters())] #Obtaining parameters
    opt = torch.optim.Adam(parameters,1e-4,betas=(0.9,0.999)) #Construct optimizers
    scheduler = lr_scheduler.OneCycleLR(opt, max_lr=1e-3, anneal_strategy='cos', total_steps=epoch_num*opt_num_epoch) #Construct schedulers for learning rate

    '''Teacher model'''
    tea_model = nn.DataParallel(UNet(3,2)).cuda() #construct teacher model
    opt_tea = torch.optim.Adam(tea_model.parameters(),1e-4,betas=(0.9,0.999))
    scheduler_tea = lr_scheduler.OneCycleLR(opt_tea, max_lr=1e-3, anneal_strategy='cos', total_steps=epoch_num*opt_num_epoch)

    '''Loading pre-trained parameters'''
    if pretrained_stu_paths is not None:
        for model_i in range(std_model_num):
            state_dict = Load_Weight_FordataParallel(torch.load(pretrained_stu_paths[model_i]),need_dataparallel=1)
            std_models[model_i].load_state_dict(state_dict) 
    if pretrained_tea_path is not None:
        state_dict = Load_Weight_FordataParallel(torch.load(pretrained_tea_path),need_dataparallel=1)
        tea_model.load_state_dict(state_dict)

    best_acc = 0
    for epoch in range(epoch_num):
        for model in std_models:
            model.train()
        tea_model.train()
        loss_mean = []
        dataloader = iter(zip(cycle(wtrain_loader), otrain_loader))

        with trange(len(otrain_loader)) as t:
            for iter_ in t:
                t.set_description('Epoch '+str(epoch))
                t.set_postfix(loss = round(np.mean(loss_mean),5))

                '''dataloading'''
                (i1_w, i2_w, label_w, _,_), (i1_o, i2_o, _,_,_) = next(dataloader)
                i1_w, i2_w, label_w = i1_w.cuda(),i2_w.cuda(),label_w.cuda().long()
                i1_o, i2_o = i1_o.cuda(),i2_o.cuda()

                '''outputting std_models'results'''
                data_w = [(model, i1_w, i2_w) for model in std_models]
                data_o = [(model, i1_o, i2_o) for model in std_models]
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    prediciton_w_stds, prediciton_o_stds = [
                        list(executor.map(infer_model, *zip(*data)))
                        for data in (data_w, data_o)
                    ]

                '''computing supervised loss'''
                prediction_w_tea = tea_model(i1_w,i2_w)
                prediciton_w_stds = torch.vstack(prediciton_w_stds)
                prediction_w = torch.cat((prediciton_w_stds,prediction_w_tea),dim=0)
                label_w = label_w.repeat(std_model_num+1,1,1)
                lossw = CE(prediction_w,label_w)
                torch.cuda.empty_cache()

                '''constructing pseudo label based on student models'''
                prediciton_o_stds = torch.stack(prediciton_o_stds)
                prediciton_o_stds = torch.softmax(prediciton_o_stds, dim=2)
                label_o_stds = torch.max(prediciton_o_stds,dim=2)[1]
                label_o_stds = torch.mean(label_o_stds.float(),dim=0)
                label_o_stds = p_yr_fun(label_o_stds)

                '''constructing pseudo label based on teacher model'''
                s,b,c,w,h = prediciton_o_stds.shape
                prediciton_o_stds = prediciton_o_stds.reshape(s*b,c,w,h)
                prediction_o_tea = torch.softmax(tea_model(i1_o,i2_o),dim=1)
                label_o_tea = torch.max(prediction_o_tea, dim=1)[1].float()

                '''computing unsupervised loss'''
                prediction_o = torch.cat((prediction_o_tea[:,1], prediciton_o_stds[:,1]),dim=0)
                label_o = torch.cat((label_o_stds, label_o_tea.repeat(std_model_num,1,1)),dim=0)
                losso = MSE(prediction_o, label_o)
                torch.cuda.empty_cache()

                '''summing losses'''
                loss = lossw + losso
                loss.backward()

                opt.step()
                opt_tea.step()
                scheduler.step(epoch=epoch-1)
                opt.zero_grad()
                opt_tea.zero_grad()
                loss_mean.append(loss.item())

        best_acc = test_tea_model(test_loader,tea_model,std_models,Log_path,epoch,best_acc)

    pretrained_stu_paths = [Log_path+'/Best_stu_model'+str(i+1)+'.pth' for i in range(std_model_num)]
    pretrained_tea_path = Log_path+'/Best_tea_model.pth'

    '''Loading pre-trained parameters'''
    if pretrained_stu_paths is not None:
        for model_i in range(std_model_num):
            state_dict = Load_Weight_FordataParallel(torch.load(pretrained_stu_paths[model_i]),need_dataparallel=1)
            std_models[model_i].load_state_dict(state_dict) 
    if pretrained_tea_path is not None:
        state_dict = Load_Weight_FordataParallel(torch.load(pretrained_tea_path),need_dataparallel=1)
        tea_model.load_state_dict(state_dict)

    '''Inference teacher and student models'''
    test_tea_model(test_loader,tea_model,std_models,Log_path,100,100)
    for model in std_models:
        test_one_model(test_loader,model,Log_path,100,100)
