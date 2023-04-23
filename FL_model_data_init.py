# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 10:09:20 2020

@author: user
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, TensorDataset
import pandas as pd
from torch.utils.data import DataLoader


def model_init(data_name):
    if data_name == 'mnist':
        model = Net_mnist()
    elif data_name == 'cifar10':
        model = Net_cifar10()
    elif data_name == 'purchase':
        model = Net_purchase()
    elif data_name == 'adult':
        model = Net_adult()
    return model


class Net_mnist(nn.Module):
    def __init__(self):
        super(Net_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.to(torch.float32)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Net_purchase(nn.Module):
    def __init__(self):
        super(Net_purchase, self).__init__()
        self.fc1 = nn.Linear(600, 300)
        self.fc2 = nn.Linear(300, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = x.to(torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x   
    

class Net_adult(nn.Module):
    def __init__(self):
        super(Net_adult, self).__init__()
        self.fc1 = nn.Linear(108, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = x.to(torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.relu(x)
        return x     


class Net_cifar10(nn.Module):
    def __init__(self):
        super(Net_cifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class All_CNN(nn.Module):
    def __init__(self, filters_percentage=1., n_channels=3, num_classes=10, dropout=False, batch_norm=True):
        super(All_CNN, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)
        self.features = nn.Sequential(
            Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),  # 14
            nn.Dropout(inplace=True) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm),
            nn.AvgPool2d(8),
            Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_filter2, num_classes),
        )
      
    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output  
    
class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
            # model += [ConvStandard(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)]
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model) 


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


"""Function: load data"""


def data_init(FL_params):
    
    kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.cuda_state else {}
    trainset, testset = data_set(FL_params.data_name)

    # train_loader = DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=True, **kwargs)
    # 构建测试数据加载器
    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=False, drop_last=True, **kwargs)

    # 将数据按照训练的trainset，均匀的分配成N-client份，所有分割得到dataset都保存在一个list中
    split_index = [int(trainset.__len__()/FL_params.N_client)]*(FL_params.N_client-1)
    split_index.append(int(trainset.__len__() - int(trainset.__len__()/FL_params.N_client)*(FL_params.N_client-1)))
    client_dataset = torch.utils.data.random_split(trainset, split_index)
    
    # 将全局模型复制N-client次，然后构建每一个client模型的优化器，参数记录
    client_loaders = []
    for ii in range(FL_params.N_client):
        client_loaders.append(DataLoader(client_dataset[ii], FL_params.local_batch_size, shuffle=False, drop_last=True, **kwargs))
        '''
        By now，我们已经将client用户的本地数据区分完成，存放在client_loaders中。每一个都对应的是某一个用户的私有数据
        '''
    
    return client_loaders, test_loader


def data_set(data_name):
    if data_name not in ['mnist', 'purchase', 'adult', 'cifar10']:
        raise TypeError('data_name should be a string, including mnist,purchase,adult,cifar10. ')
    
    # model: 2 conv. layers followed by 2 FC layers
    if data_name == 'mnist':
        trainset = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

        testset = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
        
    # model: ResNet-50
    elif data_name == 'cifar10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # model: 2 FC layers
    elif data_name == 'purchase':
        xx = np.load("data/purchase/purchase_xx.npy")
        yy = np.load("data/purchase/purchase_y2.npy")
        # yy = yy.reshape(-1,1)
        # enc = preprocessing.OneHotEncoder(categories='auto')
        # enc.fit(yy)
        # yy = enc.transform(yy).toarray()
        X_train, X_test, y_train, y_test = train_test_split(xx, yy, test_size=0.2, random_state=42)
        
        X_train_tensor = torch.Tensor(X_train).type(torch.FloatTensor)
        X_test_tensor = torch.Tensor(X_test).type(torch.FloatTensor)
        y_train_tensor = torch.Tensor(y_train).type(torch.LongTensor)
        y_test_tensor = torch.Tensor(y_test).type(torch.LongTensor)
        
        trainset = TensorDataset(X_train_tensor, y_train_tensor)
        testset = TensorDataset(X_test_tensor, y_test_tensor)

    # model: 2 FC layers
    elif data_name == 'adult':
        # load data
        file_path = "./data/adult/"
        data1 = pd.read_csv(file_path + 'adult.data', header=None)
        data2 = pd.read_csv(file_path + 'adult.test', header=None)
        data2 = data2.replace(' <=50K.', ' <=50K')    
        data2 = data2.replace(' >50K.', ' >50K')
        train_num = data1.shape[0]
        data = pd.concat([data1, data2])
       
        # data transform: str->int
        data = np.array(data, dtype=str)
        # 第15列为y
        labels = data[:, 14]
        le = LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        data = data[:, :-1]
        
        # 把类别进行数字化--LabelEncoder
        categorical_features = [1, 3, 5, 6, 7, 8, 9, 13]
        # categorical_names = {}
        for feature in categorical_features:
            le = LabelEncoder()
            le.fit(data[:, feature])
            data[:, feature] = le.transform(data[:, feature])
            # categorical_names[feature] = le.classes_
        data = data.astype(float)

        n_features = data.shape[1]
        numerical_features = list(set(range(n_features)).difference(set(categorical_features)))
        # numerical_features = [0, 2, 4, 10, 11, 12]
        # 数据归一化(0-1)
        # 格式依旧是48842行、14列
        for feature in numerical_features:
            scaler = MinMaxScaler()
            sacled_data = scaler.fit_transform(data[:, feature].reshape(-1, 1))
            data[:, feature] = sacled_data.reshape(-1)

        # 把数据进行01编码--OneHotLabel，只作用于[1, 3, 5, 6, 7, 8, 9, 13]列，其他列不变
        oh_encoder = ColumnTransformer(
            [('oh_enc', OneHotEncoder(sparse=False), categorical_features), ],
            remainder='passthrough')
        oh_data = oh_encoder.fit_transform(data)
        # 生成数据格式：48842*108，其中[0, 2, 4, 10, 11, 12]在108列的最后6列，其余列进行了01编码

        xx = oh_data
        yy = labels

        yy = np.array(yy)
        
        xx = torch.Tensor(xx).type(torch.FloatTensor)
        yy = torch.Tensor(yy).type(torch.LongTensor)
        xx_train = xx[0:data1.shape[0], :]
        xx_test = xx[data1.shape[0]:, :]
        yy_train = yy[0:data1.shape[0]]
        yy_test = yy[data1.shape[0]:]

        # trainset = Array2Dataset(xx_train, yy_train)
        # testset = Array2Dataset(xx_test, yy_test)
        trainset = TensorDataset(xx_train, yy_train)
        testset = TensorDataset(xx_test, yy_test)
        
    return trainset, testset


# define class->dataset  for adult and purchase datasets
# for the purchase, we use TensorDataset function to transform numpy.array to datasets class
# for the adult, we custom an AdultDataset class that inherits torch.util.data.Dataset class
"""
Array2Dataset: A class that can transform np.array(tensor matrix) to a torch.Dataset class.  
"""
class Array2Dataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index, :]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)
