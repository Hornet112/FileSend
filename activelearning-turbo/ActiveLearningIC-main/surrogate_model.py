#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy

import numpy as np
# import pandas as pd
# import ffx
from abc import ABC, abstractmethod
import sklearn
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from numpy import ndarray
from torch import Tensor
import torch
from botorch.models import SingleTaskGP, MultiTaskGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel, AdditiveStructureKernel
from botorch.models.transforms.outcome import Standardize
import gpytorch
import gpytorch.settings as gpts
from botorch.fit import fit_gpytorch_mll

from contextlib import ExitStack
from transform_function import *

'''
Surrogate Model Section
'''


class SurrogateBaseModel(ABC):
    def __init__(self, data_x, data_y, seed=None, transformfunction=None):
        """

        :param data_x: raw data x, tensor or ndarray, n * nx
        :param data_y: raw data y, tensor or ndarray, n * ny
        :param seed: random seed
        """

        self.train_x = data_x  # 这里考虑后续更新是加数据，因此类中应当存储原始的数据
        self.train_y = data_y
        self.n_train = data_x.shape[0]
        self.nx = data_x.shape[1]
        self.ny = data_y.shape[1]
        if seed is None:
            self.seed = 1
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        else:
            self.seed = seed
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.model = None

        self.transform = transformfunction

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def update(self, x_new, y_new):
        pass
#
#
# class FFXModel(SurrogateBaseModel):
#     def __init__(self, data_x, data_y, test_x, test_y, verbose=False, seed=None, use_gpu=False):
#         super().__init__(data_x, data_y, seed)
#         self.test_x = test_x
#         self.test_y = test_y
#         self.var_names = ["x" + str(i) for i in range(1, self.nx)]
#         self.verbose = verbose
#         if use_gpu:
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         else:
#             device = "cpu"
#         dtype = torch.double
#         self.tensor_kwargs = {"device": device, "dtype": dtype}
#         self.train_x = torch.tensor(self.train_x, **self.tensor_kwargs)
#         self.train_y = torch.tensor(self.train_y, **self.tensor_kwargs)
#         self.test_x = torch.tensor(self.test_x, **self.tensor_kwargs)
#         self.test_y = torch.tensor(self.test_y, **self.tensor_kwargs)
#
#     def train_single(self, train_y_single, test_y_single):  # 注意！没改的ffx包没有 iter_num
#         # Train and select the model with minimum test_nmse
#         train_y_single = train_y_single.numpy().flatten()
#         test_y_single = test_y_single.numpy().flatten()
#         train_x = self.train_x.numpy()
#         test_x = self.test_x.numpy()
#         models = ffx.run(train_x, train_y_single, test_x, test_y_single, self.var_names, self.verbose)
#         nmse_set = [model.test_nmse for model in models]
#         min_idx = nmse_set.index(min(nmse_set))
#         model = models[min_idx]
#         return model
#
#     def train(self):
#         if self.ny == 1:
#             self.model = self.train_single(self.train_y, self.test_y)
#         else:
#             self.model = []
#             for i in self.ny:
#                 model = self.train_single(self.train_y[:, i], self.test_y[:, i])
#                 self.model.append(model)
#
#     def train_old(self):  # 注意！没改的ffx包没有 iter_num
#         # Train and select the model with minimum test_nmse
#         models = ffx.run(self.train_x, self.train_y, self.test_x, self.test_y, self.var_names, self.verbose)
#         nmse_set = [model.test_nmse for model in models]
#         min_idx = nmse_set.index(min(nmse_set))
#         model = models[min_idx]
#         self.model = model
#
#     def update(self, x_new_train, y_new_train, x_new_test=None, y_new_test=None):
#         if self.train_x.shape[1] != x_new_train.shape[1]:
#             raise Exception('数据维数和更新前不一致')
#         self.train_x = torch.cat((self.train_x, x_new_train), dim=0)
#         self.train_y = torch.cat((self.train_y, y_new_train), dim=0)
#         print('%d points added' % x_new_train.shape[0])
#         self.n_train += x_new_train.shape[0]
#         if x_new_test is not None:
#             self.test_x = torch.cat((self.test_x, x_new_test), dim=0)
#             self.test_y = torch.cat((self.test_y, y_new_test), dim=0)
#         self.train()
#         print('The model is updated')
#
#     def predict(self, x: Tensor):
#         """
#         :param x: 1D/2D array
#         :return: y: 1D/2D array, correspondingly
#         """
#         x = x.detach().numpy()
#         if x.ndim == 1:  # single_data
#             x = x.reshape(1, len(x))
#             if self.ny == 1:
#                 y = self.model.simulate(x)
#             else:
#                 y = []
#                 for model in self.model:
#                     y.append(model.simulate(x))
#                 y = np.concatenate(y)
#         elif x.ndim == 2:
#             if self.ny == 1:
#                 y = self.model.simulate(x).reshape(-1, self.ny)
#             else:
#                 y = []
#                 for model in self.model:
#                     y.append(model.simulate(x))
#                 y = np.column_stack(y)
#         else:
#             raise Exception('不支持的输入维度')
#         y = torch.from_numpy(y)
#         return y
#
#
# class GPModel(SurrogateBaseModel):
#     """
#     def __init__(self, X_train, y_train, kernel=None):
#         self.sample_num, self.dim = X_train.shape
#         self.doe_x = torch.tensor(X_train, **tkwargs)
#         self.doe_y = torch.tensor(y_train, **tkwargs)
#         if kernel is None:
#             kernel = GPy.kern.Matern32(self.dim, ARD=True)
#         self.model = None
#         self.forget_threshold = None
#     """
#
#     def __init__(self, data_x, data_y, kernel=None, seed=None, use_gpu=False, transformfunction=None):
#         super().__init__(data_x, data_y, seed, transformfunction)
#
#         if kernel is None:
#             self.covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
#                 MaternKernel(nu=2.5, ard_num_dims=self.nx, lengthscale_constraint=Interval(0.005, 4.0))
#             )
#         else:
#             self.covar_module = kernel
#
#         if use_gpu:
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         else:
#             device = "cpu"
#         dtype = torch.double
#         self.tensor_kwargs = {"device": device, "dtype": dtype}
#         self.train_x = torch.tensor(self.train_x, **self.tensor_kwargs)
#         self.train_y = torch.tensor(self.train_y, **self.tensor_kwargs)
#
#     def train(self):
#         self.model = self.get_fitted_model()
#
#     def predict(self, x, return_cov=False):
#         """
#         This part is only valid when using Single task ! Should be modified later.
#         """
#         eeva_x = torch.tensor(x, **self.tensor_kwargs)
#
#         if self.transform is not None:
#             eeva_x = self.transform.x_transform(eeva_x, recalculate=False)
#         post = self.model(eeva_x)
#         y_mean = post.mean.reshape((-1, self.ny))
#         if self.transform is not None:
#             y_mean = self.transform.y_restore(y_mean)
#         if return_cov:
#             y_std = post.stddev.reshape((-1, self.ny))
#             if self.transform is not None:
#                 y_std = self.transform.y_std_restore(y_std)
#
#             return y_mean, y_std
#         else:
#             return y_mean
#
#     def update(self, x_new, y_new):
#
#         self.train_x = torch.cat((self.train_x, x_new), dim=0)
#         self.train_y = torch.cat((self.train_y, y_new), dim=0)
#
#         self.train()
#
#     def get_fitted_model(self):
#         likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
#
#         if self.transform is None:
#             train_x = self.train_x.detach().clone()
#             train_y = self.train_y.detach().clone()
#         else:
#             train_x, train_y = self.transform.data_transform(self.train_x, self.train_y)
#         model = SingleTaskGP(
#             train_x,
#             train_y,
#             train_Yvar=torch.full_like(train_y, 1e-8),
#
#             covar_module=self.covar_module,
#             likelihood=likelihood,
#             # outcome_transform=Standardize(m=1),
#         )
#         mll = ExactMarginalLogLikelihood(model.likelihood, model)
#         max_cholesky_size = float('inf')
#         with gpytorch.settings.max_cholesky_size(max_cholesky_size):
#             fit_gpytorch_mll(mll)
#         return model
