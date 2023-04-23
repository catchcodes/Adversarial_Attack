# -*- coding: utf-8 -*-

import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, Dataset
import copy
from sklearn.metrics import accuracy_score
import numpy as np


def FL_Train(init_global_model, client_data_loaders, test_loader, FL_params):

    # 不保存模型
    if not FL_params.save_all_models:
        print("FL Training Starting...")
        global_model = init_global_model
        for epoch in range(FL_params.global_epoch):
            client_models = global_train_once(global_model, client_data_loaders, test_loader, FL_params)
            # IMPORTANT：这里有一点要注意，就是global_train_once在训练过程中，是直接在input的client_models上进行训练，因此output的client_models与input的client_models是同一组模型，只不过input没有经过训练，而output经过了训练。
            # IMPORTANT：因此，为了实现Federated unlearning，我们需要在global train之前就将client——models中的模型进行保存。可以使用deepcopy，或者硬盘io方式。
            global_model = fedavg(client_models)
            # print(30*'^')
            # print("Global training epoch = {}".format(epoch))
            # test(global_model, test_loader)
            # print(30*'v')
        
        return global_model
    
    # 保存模型
    elif FL_params.save_all_models:
        print("FL Training Starting...")
        all_global_models = list()
        all_client_models = list()
        global_model = init_global_model
        
        all_global_models.append(copy.deepcopy(global_model))
        
        for epoch in range(FL_params.global_epoch):
            client_models = global_train_once(global_model, client_data_loaders, test_loader, FL_params)
            # IMPORTANT：这里有一点要注意，就是global_train_once在训练过程中，是直接在input的client_models上进行训练，因此output的client_models与input的client_models是同一组模型，只不过input没有经过训练，而output经过了训练。
            # IMPORTANT：因此，为了实现Federated unlearning，我们需要在global train之前就将client——models中的模型进行保存。可以使用deepcopy，或者硬盘io方式。
            global_model = fedavg(client_models)

            # print(30*'^')
            print("Global Federated Learning epoch = {}".format(epoch))
            # test(global_model, test_loader)
            # print(30*'v')

            all_global_models.append(copy.deepcopy(global_model))
            all_client_models += client_models

        return all_global_models, all_client_models
        

# 全局一轮训练，使用每一个global_model的数据、优化器，以上一轮全局模型为初始点，开始训练。
# NOTE:输入的global_model为对上一轮全局模型,输出的client_models为每一个用户分别单独训练后得到的模型。
def global_train_once(global_model, client_data_loaders, test_loader, FL_params):
    # 使用每个client的模型、优化器、数据，以client_models为训练初始模型，使用client用户本地的数据和优化器，更新得到update——client_models
    # Note：需要注意的一点是，global_train_once只是在全局上对模型的参数进行一次更新
    # update_client_models = list()
    device = torch.device("cuda" if FL_params.use_gpu*FL_params.cuda_state else "cpu")
    device_cpu = torch.device("cpu")

    client_models = []
    client_sgds = []
    for ii in range(FL_params.N_client):
        client_models.append(copy.deepcopy(global_model))
        client_sgds.append(optim.SGD(client_models[ii].parameters(), lr=FL_params.local_lr, momentum=0.9))

    for client_idx in range(FL_params.N_client):

        model = client_models[client_idx]
        optimizer = client_sgds[client_idx]

        model.to(device)
        model.train()
        
        # local training
        for local_epoch in range(FL_params.local_epoch):
            for batch_idx, (data, target) in enumerate(client_data_loaders[client_idx]):
                data = data.to(device)
                target = target.to(device)
                
                optimizer.zero_grad()
                pred = model(data)
                criteria = nn.CrossEntropyLoss()
                loss = criteria(pred, target)
                loss.backward()
                optimizer.step()
                
            if FL_params.train_with_test:
                print("Local Client No. {}, Local Epoch: {}".format(client_idx, local_epoch))
                test(model, test_loader)

        model.to(device_cpu)
        client_models[client_idx] = model

    return client_models


# 测试模型在测试集上的性能
def test(model, test_loader):
    model.eval()
    test_loss = 0
    test_acc = 0
    for data, target in test_loader:
        # print("target:", target)
        output = model(data)
        # output = torch.nn.functional.softmax(output)
        # print("output:", output)
        criteria = nn.CrossEntropyLoss()
        test_loss += criteria(output, target)  # sum up batch loss

        pred = torch.argmax(output, dim=1)
        # print("pred:", pred)
        test_acc += accuracy_score(pred, target)

    test_loss /= len(test_loader.dataset)
    test_acc = test_acc/np.ceil(len(test_loader.dataset)/test_loader.batch_size)
    # print('Test set: Average loss: {:.4f}'.format(test_loss))
    print('Test set: Average acc:  {:.4f}'.format(test_acc))    


# 联邦平均算法
def fedavg(local_models):
    """
    Parameters
    ----------
    local_models : list of local models
        DESCRIPTION. 联邦学习中，以global_model为初始模型，每一个用户使用其本地数据更新后的local model 的集合。

    Returns
    -------
    update_global_model
        使用fedavg算法更新后的全局模型
    """
    global_model = copy.deepcopy(local_models[0])
    avg_state_dict = global_model.state_dict()
    
    local_state_dicts = list()
    for model in local_models:
        local_state_dicts.append(model.state_dict())

    for layer in avg_state_dict.keys():
        avg_state_dict[layer] *= 0 
        for client_idx in range(len(local_models)):
            avg_state_dict[layer] += local_state_dicts[client_idx][layer]
        avg_state_dict[layer] /= len(local_models)
    
    global_model.load_state_dict(avg_state_dict)
    return global_model
