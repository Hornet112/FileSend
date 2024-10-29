#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import Tensor
from abc import ABC, abstractmethod

class Simulator(ABC):
    def __init__(self):
        self.sim_count = 0
        pass

    @abstractmethod
    def sim(self, x):
        pass


# class IntegratedSimulatorAlps:
#     def __init__(self, design_param, state, netlist, rootPath, bash_path, instance, key):
#         self.torch_para = {"device": 'cpu', "dtype": torch.double}
#         self.alps_simulator = AlpsInterface(netlist, rootPath, bash_path, instance)
#         self.alps_simulator.load_design()
#         self.alps_key = key
#         self.instance_num = self.alps_simulator.instance_num
#
#         x_lb = [2E-7, 2E-7, 2E-7, 1E-6, 2E-7, 2E-7,
#                 3E-7, 8E-7, 8E-7, 8E-7, 2E-7,
#                 1E-6, 1E-6, 1E-6, 7E-7, 1E-6, 1E-6,
#                 7E-7, 7E-7, 1E-6, 1E-6, 1E-6]
#         x_ub = [2E-6, 2E-6, 2E-6, 3E-6, 1E-6, 1E-6,
#                 2E-6, 3E-6, 3E-6, 3E-6, 2E-6,
#                 8E-6, 8E-6, 8E-6, 3E-6, 8E-6, 8E-6,
#                 8E-6, 8E-6, 8E-6, 8E-6, 8E-6]
#         self.x_lb = torch.DoubleTensor(x_lb).to(**self.torch_para)
#         self.x_ub = torch.DoubleTensor(x_ub).to(**self.torch_para)
#         self.sim_bounds = torch.Tensor([[110, 200], [60, 100], [2e+6, 3e+6]])
#         self.bounds = self.sim_bounds[:, 1] - self.sim_bounds[:, 0]
#
#         self.sim_num = 0
#         self.design_param = design_param
#         self.x_corner = torch.zeros(44).reshape(1, 1, -1)
#         self.state = state
#         self.anchor = None
#
#     def sim_alps(self, x, v):
#         if v.ndim == 2:
#             self.sim_num += v.shape[0]
#         else:
#             self.sim_num += v.shape[1]
#
#         try:
#             simres = self.alps_simulator.sim_with_given_param(load_design_param=False,
#                                                               design_para=x,
#                                                               mismatch=v,
#                                                               res_keys=self.alps_key
#                                                               )
#         except Exception as e:
#             raise ("出现错误：", e)
#
#         return simres
#
#     def sim_x(self, design_x):  # 优化函数，定v
#         x = self.unnorm_x(design_x)
#         output = self.sim_alps(x[0].reshape(1, -1), self.x_corner)
#         result = [[v] for v in output]
#         add = False
#         for i in range(1, x.shape[0]):
#             output = self.sim_alps(x[i].reshape(1, -1), self.x_corner)
#             for j in range(len(result)):
#                 result[j].append(output[j])
#         if self.state.sim_turbo is None:  # list(tensor(44*1)
#             self.state.sim_turbo = [(torch.cat(result[j], dim=1)).reshape(-1, 1) for j in range(len(result))]
#         else:
#             add = False
#             next_result_tensor = [(torch.cat(result[j], dim=1)).reshape(-1, 1) for j in range(len(result))]
#             self.state.sim_turbo = [torch.cat((l1, l2)) for l1, l2 in zip(self.state.sim_turbo, next_result_tensor)]
#         mc_result_tensor = torch.stack(self.state.sim_turbo).squeeze()
#         result = margin(mc_result_tensor, self.sim_bounds, self.bounds)
#         if add:
#             return result[-x.shape[0]:].reshape(-1, 1)
#         else:
#             return result.to(device='cpu', dtype=torch.double)
#
#     def ver_for_sigma(self, v):  # 分布绘制
#         mc_result_list = self.sim_alps(self.design_param, v)  # list(tensor(400*1))
#         mc_result_tensor = torch.stack(mc_result_list).squeeze()
#         result = margin(mc_result_tensor, self.sim_bounds, self.bounds)
#         return result.reshape(-1)
#
#     def sim_v(self, v):  # 角提取，定x
#         mc_result_list = self.sim_alps(self.design_param, v)  # list(tensor(400*1))
#         if self.state.sim_output is None:
#             self.state.sim_output = mc_result_list
#         else:
#             self.state.sim_output = [torch.cat((l1, l2)) for l1, l2 in zip(self.state.sim_output, mc_result_list)]
#         mc_result_tensor = torch.stack(self.state.sim_output).squeeze()
#         result = margin(mc_result_tensor, self.sim_bounds, self.bounds)
#
#         return result
#
#     def turbo_anchor(self, design_x):  # 优化过程中计算锚点值
#         x = self.unnorm_x(design_x)
#         anchor_turbo_v_list = []
#         for t in range(0, len(self.anchor)):
#             output = self.sim_alps(x, self.anchor[t])
#             total_result = [torch.cat((l1, l2)) for l1, l2 in zip(self.state.sim_turbo, output)]
#             mc_result_tensor = torch.stack(total_result).squeeze()
#             anchor_turbo_v_list.append(margin(mc_result_tensor, self.sim_bounds, self.bounds)[-1:])
#         anchor_turbo_v_tensor = torch.cat(anchor_turbo_v_list, dim=0)
#         dis = [(anchor_turbo_v_tensor[j] - anchor_turbo_v_tensor[j - 1]).item()
#                for j in range(1, len(anchor_turbo_v_tensor))]
#         return dis
#
#     def norm_x(self, x):
#         return (x - self.x_lb) / (self.x_ub - self.x_lb)
#
#     def unnorm_x(self, x):
#         return x * (self.x_ub - self.x_lb) + self.x_lb


class IntegratedSimulatorFunction:
    def __init__(self, function, x_ub, x_lb):
        self.sim_count = 0
        self.fun_handle = function
        self.v_corner = None
        self.design_x = None

        self.x_ub = x_ub
        self.x_lb = x_lb
        pass

    def sim_x(self, x_norm, v_corner=None):
        x = self.unnorm_x(x_norm)
        if v_corner is None:
            aug_sample = self.tensor_aug_sample(x, self.v_corner)[0]
        else:
            aug_sample = self.tensor_aug_sample(x, v_corner)[0]

        self.sim_count += aug_sample.shape[0]
        return self.fun_handle(aug_sample)

    def sim_v(self, v, design_x=None):
        if design_x is None:
            x = self.unnorm_x(self.design_x)
            aug_sample = self.tensor_aug_sample(x, v)[0]
        else:
            x = self.unnorm_x(design_x)
            aug_sample = self.tensor_aug_sample(x, v)[0]

        self.sim_count += aug_sample.shape[0]
        return self.fun_handle(aug_sample)

    def unnorm_x(self, x):
        if self.x_lb is None or self.x_ub is None:
            raise Exception('x_lb or x_ub undefined')
        return x * (self.x_ub - self.x_lb) + self.x_lb

    @staticmethod
    def tensor_aug_sample(x: Tensor, v: Tensor):
        n_x = x.shape[0]
        n_v = v.shape[0]
        inx_x = torch.linspace(0, n_x - 1, n_x).reshape((-1, 1))
        inx_v = torch.linspace(0, n_v - 1, n_v).reshape((-1, 1))
        x_part = torch.repeat_interleave(x, n_v, dim=0)
        v_part = v.repeat(n_x, 1)
        sample = torch.cat((x_part, v_part), dim=1)
        ind_aug = torch.cat((torch.repeat_interleave(inx_x, n_v, dim=0), inx_v.repeat(n_x, 1)), dim=1)
        return sample, ind_aug
