import torch
import random
import copy
import numpy as np
import matplotlib.pyplot as plt

from FL_base_function import test
from AdvBox.adversarialbox.adversary import Adversary
from AdvBox.adversarialbox.attacks.deepfool import DeepFoolAttack
from AdvBox.adversarialbox.models.pytorch import PytorchModel
from attack import vceg

def inference_attack(all_GMs, train_loader, test_loader, device, FL_params):
    # 为11*11*20矩阵((特征数+1)*(训练轮数+1)*组数)
    train_input = []
    test_input = []
    # 取多组数据
    for each_attack_epoch in range(FL_params.attack_epoch):
        # 生成一个0-64随机数
        data = random.randint(0, 63)
        # 时间轴数组，存放每个数据随着训练轮数变化，距离值的变化
        # 最后为11*11矩阵((特征数+1)*(训练轮数+1))
        Distance_train = []
        Distance_test = []
        # 对联邦学习过程中的每次全局模型进行分析
        for each_epoch in range(FL_params.global_epoch + 1):
            # 目标模型
            model = all_GMs[each_epoch]

            # 打印选取目标模型在测试集上的准确率
            print("attack_epoch: {}, round: {}".format(each_attack_epoch + 1, each_epoch))
            if not each_attack_epoch:
                test(model, test_loader)

            model = model.to(device)
            model = model.eval()

            # 设置为不保存梯度值 自然也无法修改
            for param in model.parameters():
                param.requires_grad = False

            # input_min = test_loader.sampler.data_source.tensors[0].min()
            # input_max = test_loader.sampler.data_source.tensors[0].max()
            # bounds = (input_min, input_max)
            # M_tgt = PytorchModel(model, None, bounds, channel_axis=1, nb_classes=2)
            #
            # deepfool_attacker = DeepFoolAttack(M_tgt)
            # attacker_config = {"iterations": 500, "overshoot": 0.02}

            print("train_predict...")

            # distance_train存放的是每一轮中各个数据点的距离值
            distance_train = []
            # 进入主循环(训练集)
            for _, (XX_tgt, YY_tgt) in enumerate(train_loader):

                # XX_tgt.shape = [64, 108] -> [1, 108]
                # 在一批数据(64)中随机取一个数据点
                XX_tgt = XX_tgt[data, :]
                XX_tgt = XX_tgt.unsqueeze(0)
                XX_tgt = XX_tgt.cpu().numpy()

                # 生成一个扰动样本作为测试样本
                XX_tgt_new_1 = copy.deepcopy(XX_tgt)
                XX_tgt_new_1[0, 102] += 0.2
                XX_tgt_new_2 = copy.deepcopy(XX_tgt)
                XX_tgt_new_2[0, 0:9] = [1, 0, 0, 0, 0, 0, 0, 0, 0]
                XX_tgt_new_3 = copy.deepcopy(XX_tgt)
                XX_tgt_new_3[0, 103] += 0.1
                XX_tgt_new_4 = copy.deepcopy(XX_tgt)
                XX_tgt_new_4[0, 9:25] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                XX_tgt_new_5 = copy.deepcopy(XX_tgt)
                XX_tgt_new_5[0, 104] += 0.5
                XX_tgt_new_6 = copy.deepcopy(XX_tgt)
                XX_tgt_new_6[0, 25:32] = [1, 0, 0, 0, 0, 0, 0]
                XX_tgt_new_7 = copy.deepcopy(XX_tgt)
                XX_tgt_new_7[0, 32:47] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                XX_tgt_new_8 = copy.deepcopy(XX_tgt)
                XX_tgt_new_8[0, 47:53] = [0, 1, 0, 0, 0, 0]
                XX_tgt_new_9 = copy.deepcopy(XX_tgt)
                XX_tgt_new_9[0, 53:58] = [1, 0, 0, 0, 0]
                XX_tgt_new_10 = copy.deepcopy(XX_tgt)
                XX_tgt_new_10[0, 58:60] = [1, 0]
                list_label = [XX_tgt, XX_tgt_new_1, XX_tgt_new_2, XX_tgt_new_3,
                              XX_tgt_new_4, XX_tgt_new_5, XX_tgt_new_6, XX_tgt_new_7,
                              XX_tgt_new_8, XX_tgt_new_9, XX_tgt_new_10]

                YY_tgt = None

                for each_XX_tgt in list_label:
                    each_XX_tgt = torch.from_numpy(each_XX_tgt)
                    advs = vceg(model, each_XX_tgt)
                    d = torch.sqrt(torch.sum((torch.from_numpy(XX_tgt) - advs) ** 2))
                    # adversary = Adversary(each_XX_tgt, YY_tgt)
                    # adversary = deepfool_attacker(adversary, **attacker_config)
                    #
                    # if adversary.is_successful():
                    #     advs = adversary.adversarial_example[0]
                    #
                    #     # 对抗成功的最小的扰动值
                    #     d = torch.sqrt(torch.sum((torch.from_numpy(XX_tgt) - torch.from_numpy(advs)) ** 2))
                    #
                    #     print("attack success, adv_label={}, distance={}".format(adversary.adversarial_label, d))
                    #
                    # else:
                    #     advs = adversary.bad_adversarial_example[0]
                    #
                    #     d = torch.sqrt(torch.sum((torch.from_numpy(XX_tgt) - torch.from_numpy(advs)) ** 2))
                    #
                    #     print("attack failed, distance=", d)
                    print(f"distance= {d}")
                    distance_train.append(d)

                # 只进行一轮
                break

            print("test_predict...")

            distance_test = []
            # 进入主循环(测试集)
            for _, (XX_tgt, YY_tgt) in enumerate(test_loader):

                # XX_tgt.shape = [64, 108] -> [1, 108]
                # 在一批数据(64)中随机取一个数据点
                XX_tgt = XX_tgt[data, :]
                XX_tgt = XX_tgt.unsqueeze(0)
                XX_tgt = XX_tgt.cpu().numpy()

                # 生成一个扰动样本作为测试样本
                XX_tgt_new_1 = copy.deepcopy(XX_tgt)
                XX_tgt_new_1[0, 102] += 0.2
                XX_tgt_new_2 = copy.deepcopy(XX_tgt)
                XX_tgt_new_2[0, 0:9] = [1, 0, 0, 0, 0, 0, 0, 0, 0]
                XX_tgt_new_3 = copy.deepcopy(XX_tgt)
                XX_tgt_new_3[0, 103] += 0.1
                XX_tgt_new_4 = copy.deepcopy(XX_tgt)
                XX_tgt_new_4[0, 9:25] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                XX_tgt_new_5 = copy.deepcopy(XX_tgt)
                XX_tgt_new_5[0, 104] += 0.5
                XX_tgt_new_6 = copy.deepcopy(XX_tgt)
                XX_tgt_new_6[0, 25:32] = [1, 0, 0, 0, 0, 0, 0]
                XX_tgt_new_7 = copy.deepcopy(XX_tgt)
                XX_tgt_new_7[0, 32:47] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                XX_tgt_new_8 = copy.deepcopy(XX_tgt)
                XX_tgt_new_8[0, 47:53] = [0, 1, 0, 0, 0, 0]
                XX_tgt_new_9 = copy.deepcopy(XX_tgt)
                XX_tgt_new_9[0, 53:58] = [1, 0, 0, 0, 0]
                XX_tgt_new_10 = copy.deepcopy(XX_tgt)
                XX_tgt_new_10[0, 58:60] = [1, 0]
                list_label = [XX_tgt, XX_tgt_new_1, XX_tgt_new_2, XX_tgt_new_3,
                              XX_tgt_new_4, XX_tgt_new_5, XX_tgt_new_6, XX_tgt_new_7,
                              XX_tgt_new_8, XX_tgt_new_9, XX_tgt_new_10]

                # list_label = feature_extract(XX_tgt, FL_params.n_feature)

                YY_tgt = None

                for each_XX_tgt in list_label:
                    #
                    # adversary = Adversary(each_XX_tgt, YY_tgt)
                    # adversary = deepfool_attacker(adversary, **attacker_config)
                    #
                    # if adversary.is_successful():
                    #     advs = adversary.adversarial_example[0]
                    #
                    #     # 对抗成功的最小的扰动值
                    #     d = torch.sqrt(torch.sum((torch.from_numpy(XX_tgt) - torch.from_numpy(advs)) ** 2))
                    #     print("attack success, adv_label={}, distance={}".format(adversary.adversarial_label, d))
                    #
                    # else:
                    #     advs = adversary.bad_adversarial_example[0]
                    #     d = torch.sqrt(torch.sum((torch.from_numpy(XX_tgt) - torch.from_numpy(advs)) ** 2))
                    #     print("attack failed, distance=", d)
                    each_XX_tgt = torch.from_numpy(each_XX_tgt)
                    advs = vceg(model, each_XX_tgt)
                    d = torch.sqrt(torch.sum((torch.from_numpy(XX_tgt) - advs) ** 2))
                    print(f"distance= {d}")
                    distance_test.append(d)

                # 只进行一轮
                break

            Distance_train.append(distance_train)
            Distance_test.append(distance_test)

        # 进行转置，便于观察
        Distance_train = [[row[i] for row in Distance_train] for i in range(len(Distance_train[0]))]
        Distance_test = [[row[i] for row in Distance_test] for i in range(len(Distance_test[0]))]

        # display_image(Distance_train, "train")
        # display_image(Distance_test, "test")

        train_input.append(Distance_train)
        test_input.append(Distance_test)

    train_input = torch.tensor(train_input)
    test_input = torch.tensor(test_input)

    torch.set_printoptions(threshold=np.inf)

    print("train_input:", train_input)
    print("test_input:", test_input)

    return train_input, test_input


# 特征提取函数，对原数据进行扰动
def feature_extract(XX_tgt, n):
    # 定义每个特征可以扰动的个数，共66个，即此时n_max=66
    N_feature = [0, 2, 9, 2, 16, 2, 7, 15, 6, 5, 2]
    XX_tgt_new = []

    # 修改特征1
    for xx1 in range(N_feature[1]):
        # 生成一个0.2-0.5的随机数
        seed_1 = random.randint(20, 50) / 100
        xx = copy.deepcopy(XX_tgt)
        xx[0, 102] += seed_1
        XX_tgt_new.append(xx)

    # 修改特征2
    seed_2 = [1, 0, 0, 0, 0, 0, 0, 0, 0]
    for xx2 in range(N_feature[2] - 1):
        # 定义修改的特征情况
        xx = copy.deepcopy(XX_tgt)
        xx[0, 0:9] = seed_2
        # 列表右移一位
        seed_2.insert(0, seed_2.pop())
        XX_tgt_new.append(xx)

    # 修改特征3
    for xx3 in range(N_feature[3]):
        # 生成一个0.05-0.2的随机数
        seed_3 = random.randint(5, 20) / 100
        xx = copy.deepcopy(XX_tgt)
        xx[0, 103] += seed_3
        XX_tgt_new.append(xx)

    # 修改特征4
    seed_4 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for xx4 in range(N_feature[4] - 1):
        # 定义修改的特征情况
        xx = copy.deepcopy(XX_tgt)
        xx[0, 9:25] = seed_4
        # 列表右移一位
        seed_4.insert(0, seed_4.pop())
        XX_tgt_new.append(xx)

    # 修改特征5
    for xx5 in range(N_feature[5]):
        # 生成一个0.4-0.6的随机数
        seed_5 = random.randint(40, 60) / 100
        xx = copy.deepcopy(XX_tgt)
        xx[0, 104] += seed_5
        XX_tgt_new.append(xx)

    # 修改特征6
    seed_6 = [1, 0, 0, 0, 0, 0, 0]
    for xx6 in range(N_feature[6] - 1):
        # 定义修改的特征情况
        xx = copy.deepcopy(XX_tgt)
        xx[0, 25:32] = seed_6
        # 列表右移一位
        seed_6.insert(0, seed_6.pop())
        XX_tgt_new.append(xx)

    # 修改特征7
    seed_7 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for xx7 in range(N_feature[7] - 1):
        # 定义修改的特征情况
        xx = copy.deepcopy(XX_tgt)
        xx[0, 32:47] = seed_7
        # 列表右移一位
        seed_7.insert(0, seed_7.pop())
        XX_tgt_new.append(xx)

    # 修改特征8
    seed_8 = [1, 0, 0, 0, 0, 0]
    for xx8 in range(N_feature[8] - 1):
        # 定义修改的特征情况
        xx = copy.deepcopy(XX_tgt)
        xx[0, 47:53] = seed_8
        # 列表右移一位
        seed_8.insert(0, seed_8.pop())
        XX_tgt_new.append(xx)

    # 修改特征9
    seed_9 = [1, 0, 0, 0, 0]
    for xx9 in range(N_feature[9] - 1):
        # 定义修改的特征情况
        xx = copy.deepcopy(XX_tgt)
        xx[0, 53:58] = seed_9
        # 列表右移一位
        seed_9.insert(0, seed_9.pop())
        XX_tgt_new.append(xx)

    # 修改特征10
    seed_10 = [1, 0]
    for xx10 in range(N_feature[10] - 1):
        # 定义修改的特征情况
        xx = copy.deepcopy(XX_tgt)
        xx[0, 58:60] = seed_10
        # 列表右移一位
        seed_10.insert(0, seed_10.pop())
        XX_tgt_new.append(xx)

    # 从生成的所有特征中随机取n个元素组成新的列表
    XX_tgt_new = random.sample(XX_tgt_new, n)
    # 添加原数据到列表首元素
    XX_tgt_new.append(XX_tgt)
    XX_tgt_new.insert(0, XX_tgt_new.pop())

    return XX_tgt_new


# 显示图象
def display_image(image, mode):
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    plt.figure(figsize=(20, 15))

    for i in range(len(image)):
        plt.plot(x, image[i], '-o', label=mode+" property "+str(i), linewidth=2.0)

    plt.legend()  # 显示图例，如果注释改行，即使设置了图例仍然不显示
    plt.show()


# 测试训练模型精度专用，得到一个batch
def for_test(all_GMs, train_loader, test_loader, device, FL_params):
    # 目标模型
    for i in range(FL_params.global_epoch):
        model = all_GMs[i]

        # 打印选取目标模型在测试集上的准确率
        test(model, train_loader)

        print("**********************************")

        test(model, test_loader)

