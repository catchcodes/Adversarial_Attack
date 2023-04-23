import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.nn import init

from attack_function import inference_attack


# 定义孪生网络
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            # nn.ReLU(inplace=True),
            nn.Hardswish(inplace=True),
            nn.BatchNorm2d(4),
            # nn.BatchNorm2d(4)中参数4为通道数
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            # nn.ReLU(inplace=True),
            nn.Hardswish(inplace=True),
            nn.BatchNorm2d(8),
            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            # nn.ReLU(inplace=True),
            nn.Hardswish(inplace=True),
            nn.BatchNorm2d(8),

        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 10 * 21, 50),
            # nn.ReLU(inplace=True),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),

            nn.Linear(50, 50),
            nn.Dropout(p=0.1),
            nn.Hardswish(inplace=True),

            nn.Linear(50, 2)
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()


    def forward_once(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        output = self.cnn1(x)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


# 定义损失函数
class ContrastiveLoss(nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    # 使用pairwise_distance计算欧式距离后，使用对比损失作为目标损失函数 margin:阈值
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


# 初始化孪生网络数据集
def Siamese_data_init(input_train, input_test):
    # 合并数据集
    input_X = list(torch.cat((input_train, input_test), 0))
    input_y = []
    # 1为相同类型，0为不同类型
    for _ in input_train:
        input_y.append(0)
    for _ in input_test:
        input_y.append(1)

    # 将数据集划分为训练集和测试集，返回X最外层为list，内层均为tensor；y为list类型
    X_train, X_test, y_train, y_test = train_test_split(input_X, input_y, test_size=0.6, shuffle=True, random_state=7)

    # 得到数据个数
    number_train = len(y_train)
    number_test = len(y_test)

    # 将训练集和测试集分为两组数据,加入噪声增强差异性（待证实）
    # 训练集：同一X[(d0,……,d0)T, (d1,……,d10)T, 1(input_train) or 0(input_test)]
    train_list = []
    i = 0
    for each_input in X_train:
        data1 = each_input[0].unsqueeze(0)
        data2 = each_input[1].unsqueeze(0)
        for ii in range(len(each_input) - 2):
            data1 = torch.cat((data1, each_input[0].unsqueeze(0)), 0)
            data2 = torch.cat((data2, each_input[ii + 2].unsqueeze(0)), 0)

        train_list.append([data1, data2, y_train[i]])
        i += 1

    test_list = []
    i = 0
    # 测试集：同一X[(d0,……,d0)T, (d1,……,d10)T, 1(input_train) or 0(input_test)]
    for each_input in X_test:
        data1 = each_input[0].unsqueeze(0)
        data2 = each_input[1].unsqueeze(0)
        for ii in range(len(each_input) - 2):
            data1 = torch.cat((data1, each_input[0].unsqueeze(0)), 0)
            data2 = torch.cat((data2, each_input[ii + 2].unsqueeze(0)), 0)

        test_list.append([data1, data2, y_test[i]])
        i += 1

    ##############################################################################
    # 不同X，数据增强
    DistanceTrain = []
    DistanceTest = []
    TrainSampleNumber = 0
    TestSampelNumber = 0
    for ReList in train_list:
        if ReList[2] == 1:
            DistanceTrain.append(ReList[0])
            TrainSampleNumber += 1
        if ReList[2] == 0:
            DistanceTest.append(ReList[0])
            TestSampelNumber += 1
    AddSampleNumber = min(TrainSampleNumber, TestSampelNumber)
    number_train += AddSampleNumber
    for i in range(AddSampleNumber):
        train_list.append([DistanceTrain[i], DistanceTest[i], 0])

    DistanceTrainInTest = []
    DistanceTestInTest = []
    TrainSampleNumberInTest = 0
    TestSampelNumberInTest = 0
    for ReList in test_list:
        if ReList[2] == 1:
            DistanceTrainInTest.append(ReList[0])
            TrainSampleNumberInTest += 1
        if ReList[2] == 0:
            DistanceTestInTest.append(ReList[0])
            TestSampelNumberInTest += 1
    AddSampleNumberInTest = min(TrainSampleNumberInTest, TestSampelNumberInTest)
    number_test += AddSampleNumberInTest
    for i in range(AddSampleNumberInTest):
        test_list.append([DistanceTrainInTest[i], DistanceTestInTest[i], 0])
    ##############################################################################

    return train_list, test_list, number_train, number_test


# 训练孪生网络
def Siamese_function(input_train, input_test):
    # torch.manual_seed(114514)
    # 孪生网络数据集初始化
    train_list, test_list, number_train, number_test = Siamese_data_init(input_train, input_test)

    # 初始化孪生网络、损失函数、优化器
    siamese_net = SiameseNetwork()
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(siamese_net.parameters(), 0.002, betas=(0.9, 0.999))
    # 动态学习率，每隔50epoch衰减50%
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

    print("siamese_net training start...")

    similarity_train = []
    similarity_test = []

    for epoch in range(1, 201):
        running_loss = 0
        # distance0为距离列表第一行构成的矩阵、distance1为剩余行构成的矩阵
        # label为是否相似（即同为测试集/一个属于训练集一个属于测试集）
        for distance0, distance1, label in train_list:
            optimizer.zero_grad()

            # 将二维转化为四维，即为1*1*10*11
            distance0 = distance0.unsqueeze(0).unsqueeze(0)
            distance1 = distance1.unsqueeze(0).unsqueeze(0)

            output1, output2 = siamese_net(distance0, distance1)
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            # scheduler.step()

            running_loss += loss_contrastive.item()

        print("epoch: {}, loss: {}".format(epoch, running_loss))

        # 攻击模型测试
        if epoch % 1 == 0:
            score_label = []
            score_label_predict = []
            p_distance = []
            for distance0, distance1, label in train_list:
                distance0 = distance0.unsqueeze(0).unsqueeze(0)
                distance1 = distance1.unsqueeze(0).unsqueeze(0)

                output1, output2 = siamese_net(distance0, distance1)
                euclidean_distance = F.pairwise_distance(output1, output2)
                if euclidean_distance > 2:
                    euclidean_distance = 2
                p_distance.append(euclidean_distance)
                if euclidean_distance > 0.5:
                    # 认为不相似
                    label_predict = 1
                else:
                    label_predict = 0

                score_label.append(label)
                score_label_predict.append(label_predict)

            print(f"accuracy_train :  {round(accuracy_score(score_label, score_label_predict) * 100, 2)}%")

            similarity_train.append(p_distance)

        # 攻击模型测试
        if epoch % 1 == 0:
            score_labe = []
            score_labe_predict = []
            p_distanc = []
            for distance0, distance1, label in test_list:
                distance0 = distance0.unsqueeze(0).unsqueeze(0)
                distance1 = distance1.unsqueeze(0).unsqueeze(0)

                output1, output2 = siamese_net(distance0, distance1)
                euclidean_distance = F.pairwise_distance(output1, output2)
                if euclidean_distance > 2:
                    euclidean_distance = 2
                p_distanc.append(euclidean_distance)
                if euclidean_distance > 0.5:
                    # 认为不相似
                    label_predict = 1
                else:
                    label_predict = 0

                score_labe.append(label)
                score_labe_predict.append(label_predict)

            print(f"accuracy_test :  {round(accuracy_score(score_labe, score_labe_predict) * 100, 2)}%")

            similarity_test.append(p_distanc)


import matplotlib.pyplot as plt


# 显示图象
def display_image(image, number,yLabel):
    # 设置横轴，为训练时刻数
    x = []
    for i in range(200):
        x.append(i)

    plt.figure(figsize=(200, 150))
    plt.ylabel(yLabel)


    # 设置纵轴，为相似度变化
    for i in range(number):
        y = [k[i] for k in image]
        plt.plot(x, y, '-o', label=str(i), linewidth=1.0)

    plt.legend()
    plt.show()


# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)

