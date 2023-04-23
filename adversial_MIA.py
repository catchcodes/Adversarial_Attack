# -*- coding: utf-8 -*-
"""
针对联邦学习，基于对抗样本技术的成员推断攻击。
"""

import numpy as np
import torch

# self defined function
from FL_base_function import FL_Train
from FL_model_data_init import model_init, data_init
from attack_function import inference_attack
from Siamese_Network import Siamese_function

newDistance = False


class Arguments():
    def __init__(self):
        # Federated Learning Settings
        self.N_client = 5
        self.data_name = 'adult'  # purchase, cifar10, mnist, adult
        self.global_epoch = 20
        self.local_epoch = 2

        # Local Model Training Settings
        self.local_batch_size = 64
        self.local_lr = 0.01

        # 攻击模型的数据组数
        self.attack_epoch = 40


        # 取的特征个数
        self.n_feature = 10

        self.test_batch_size = self.local_batch_size
        self.seed = 1
        self.save_all_models = True
        self.cuda_state = False
        self.use_gpu = False
        self.train_with_test = True

        # MIA settings
        self.target_client = np.random.randint(0, self.N_client)


def adv_MIA():
    global input_train, input_test
    if newDistance:
        """1. Initialize FL Settings, global model, and data loaders"""
        # training settings
        FL_params = Arguments()
        # GM->global model  CM->client model
        # init_GM --> initial global model
        init_GM = model_init(FL_params.data_name)
        # client_loaders -> For each local client, list of instances of torch.utils.data.Dataloader
        # test_loader -> A global dataset for testing, an instance of torch.utils.data.Dataloader
        client_loaders, test_loader = data_init(FL_params)

        """2. Training Process for Federated Learning"""
        # all_GMs -> init GM, and list of instances of global models from epoch=0 ~ T，
        # all_LMs -> list of instances of local models from epoch=0 ~ T for each client
        all_GMs, all_LMs = FL_Train(init_GM, client_loaders, test_loader, FL_params)

        """3. Constructing Adversial Examples for the training data """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 目标用户的数据加载器 Dataloader
        data_loader_tgt = client_loaders[FL_params.target_client]
        dataset_tgt = data_loader_tgt.sampler.data_source

        """4. Computing the Distance between Adversial Examples and Original Examples """
        # distance -> the Distance between Adversial Examples and Original Examples
        # correct -> the prediction is (or not) correct
        # train_label, train_distances, train_correct, test_label, test_distances, test_correct \
        #     = deepfool_attack(all_GMs, data_loader_tgt, test_loader, device, FL_params)
        input_train, input_test = inference_attack(all_GMs, data_loader_tgt, test_loader, device, FL_params)
        torch.save(input_train, "Distance_vceg\\input_train_distance_" + FL_params.data_name + "3.pth")
        torch.save(input_test, "Distance_vceg\\input_test_distance_" + FL_params.data_name + "3.pth")
    else:
        FL_params = Arguments()
        try:
            input_train = torch.load("Distance_vceg\\input_train_distance_" + FL_params.data_name + "2.pth")
            input_test = torch.load("Distance_vceg\\input_test_distance_" + FL_params.data_name + "2.pth")
        except FileNotFoundError:
            print("*" * 60 + "\nERROR : 没有" + FL_params.data_name + "相应模型的边界距离!\n" + "*" * 60)

    """5. Constructing the Attack Distance for Local Distance """
    Siamese_function(input_train, input_test)

    # for_test(all_GMs, data_loader_tgt, test_loader, device, FL_params)


if __name__ == "__main__":
    adv_MIA()
