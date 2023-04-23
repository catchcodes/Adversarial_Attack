#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :attack.py
# @Time      :2023/2/25 17:05
# @Author    :Ouyang Bin
import copy
import numpy as np
import torch


def vceg(model, sample, target_label=None, target=None, gamma=0.5,
         bin_theta=2.0, iteration=200, num_evals=500, delta=0.1):
    """
	Attack method of gradient estimation based on vector synthesis
	Input:
		model  : 模型
		sample : 样本
		target_label : 定向攻击标签（待实现）
		target : 定向攻击对象（待实现）
		gamma  : 合成向量的约束
    	bin_theta : bin-search阈值
    	iteration : 合成向量寻找距离的迭代次数
    	num_evals : approximate_gradient中单位向量的个数
    	delta     : approximate_gradient中单位向量的约束

	Output:
		distance : sample样本到模型决策边界的距离
	"""
    sample_label = predict(model, sample)
    sample_d = find_sample_d(model, sample, sample_label)

    dimension = int(np.prod(sample.shape))
    # 二分的阈值theta = bin_theta / 样本维数**(2-1/p) p=0,2,inf , 本算法使用l2
    theta = bin_theta / (np.sqrt(dimension) * dimension)
    eps = 1e-16

    x_boundary = bin_search(model, sample, sample_d, sample_label, theta)
    x_d_next = x_boundary + gamma*approximate_gradient(model, sample, sample_label, num_evals, delta)

    # 迭代iteration（默认100）轮
    for i in range(iteration):
        x_boundary_next = bin_search(model, sample, x_d_next, sample_label, theta)
        x_boundary_sym = 2 * x_boundary_next - x_boundary
        x_boundary_nn = bin_search(model, sample, x_boundary_sym, sample_label, theta)
        # 凸边界
        if predict(model, x_boundary_sym) != sample_label:
            composite_vector = 2*x_boundary_next-x_boundary_nn-x_boundary
            composite_vector /= (np.linalg.norm(composite_vector.numpy())+eps)  # 防止出现/0计算
            x_d_nn = x_boundary_next + gamma*(composite_vector)
        # 凹边界
        else:
            composite_vector = x_boundary+x_boundary_nn-2*x_boundary_next
            composite_vector /= (np.linalg.norm(composite_vector.numpy())+eps)  # 防止出现/0计算
            x_d_nn = x_boundary_next + gamma*(composite_vector)

        sample_d = x_d_next
        x_d_next = x_d_nn
        x_boundary = bin_search(model, sample, sample_d, sample_label, theta)

    boundary = bin_search(model, sample, x_d_next, sample_label, theta)
    # distance = torch.sqrt(torch.sum((boundary - sample)**2))

    return boundary


# 二分查找边界上的点
def bin_search(model, x, x_d, x_label, threshold):
    num_evals = 0
    x_origin = copy.deepcopy(x_d)
    # x和x_d同类，出现在边界为凹时，会出现找不到边界点的情况
    while (predict(model, x_d) == x_label):
        x_d = 2 * x_d - x
        num_evals += 1
        if num_evals > 2:
            # assert num_evals<=3, "找不到边界点"
            return x_origin  # 用一开始的x_d代替边界点，以当前距离作为最终距离（**待优化**）
    # assert predict(model, x_d) != x_label, "bin-search不同类"

    distance = np.linalg.norm(x - x_d)
    while distance > threshold:
        if predict(model, (x + x_d)/2) == x_label:
            x = (x + x_d) / 2
        else:
            x_d = (x + x_d) / 2
        distance = np.linalg.norm(x - x_d)
    # 边界点可能是和sample标签一致的点，也可能是不一致的点（没多大影响）
    return (x + x_d) / 2


# 查询模型
def predict(model, x):
    y = model(x)
    y.detach()
    return torch.argmax(y)


# 找到一个与sample样本不同类的sample_d
def find_sample_d(model, sample, sample_label):
    num_evals = 0
    while True:
        random_noise = torch.randn(sample.shape)
        random_noise = random_noise.uniform_(-2, 2)  # 可以适当调整范围，让寻找更容易
        success = (predict(model, random_noise) != sample_label)
        num_evals += 1
        if success:
            break
        # assert num_evals < 1e4, "未找到不同类的样本！"  # 有较小概率会出现这种情况(**待优化**)

    # 减小sample_d和sample的差异（距离）
    low = 0.0
    high = 1.0
    while high - low > 0.001:
        mid = (high + low) / 2.0
        blend = (1 - mid) * sample + mid * random_noise
        success = (predict(model, blend) != sample_label)
        if success:
            high = mid
        else:
            low = mid
    # 一开始high为1，sample_d还是产生的噪声，high改变时为success为1时（predict(model, blend) != sample_label），blend必为异类
    sample_d = (1 - high) * sample + high * random_noise
    # 假距离问题（局部最优），sample点和决策边界有两个垂直距离
    # 初始点落在哪边，哪边的距离就是迭代出的距离，但未必是最短距离
    # 用随机初始化的异类点的对称点作为初始点，在边界平滑的情况可以解决
    # sample_d_sym = 2*sample - sample_d（待完善）

    return sample_d


# 近似梯度作为初始边界点移动方向，一个数据仅调用一次
def approximate_gradient(model, sample, sample_label, num_evals, delta):
    # Generate random vectors.
    noise_shape = [num_evals] + list(sample.shape)
    ub = np.random.randn(*noise_shape)
    eps = 1e-16
    # 各个方向的单位向量(num_evals个)   Adult:axis=(1,2)  cifar-10:axis=(1, 2, 3)
    ub = ub / (np.sqrt(np.sum(ub ** 2, axis=(1, 2), keepdims=True))+eps)  # 保持维度不变
    weight = []
    ub = torch.from_numpy(ub)
    for vector in ub:
        weight.append(int(predict(model, vector*delta)!=sample_label))

    weight[:] = [2*x-1 for x in weight]

    # # baseline 效果更差(**待找到原因**)
    # weight = np.array(weight)
    # average = (1/num_evals)*np.sum(weight)
    # weight = weight - average

    composite_vector = torch.zeros(sample.shape)
    for i in range(len(weight)):
        composite_vector = composite_vector + ub[i] * weight[i]

    gradf = (1/(num_evals))*composite_vector
    gradf = gradf / (np.linalg.norm(gradf)+eps)

    return gradf


# 第一个边界点的移动 找一个距离sample更近且不同类的点
# def initial_move(model, sample, sample_label, x_boundary, gamma):
#     num_evals = 0
#
#     while True:
#         x_boundary_move = gamma * torch.randn(size=sample.shape) + x_boundary
#         success = (predict(model, x_boundary_move) != sample_label) and \
#                   (np.linalg.norm(sample - x_boundary_move) < np.linalg.norm(sample - x_boundary))
#         num_evals += 1
#         if success:
#             break
#         if num_evals > 1e4:
#             global fail
#             fail = True
#             break
#
#     return x_boundary_move