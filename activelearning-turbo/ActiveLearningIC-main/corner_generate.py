import copy

import numpy as np
import torch
from torch import Tensor
from abc import ABC, abstractmethod
from torch.quasirandom import SobolEngine
from simulator import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize

'''
Yield Section
'''


class CornerGenerate(ABC):
    def __init__(self, simulator: Simulator, mean: Tensor, std: Tensor, random_generator: SobolEngine, use_gpu=False):
        self.simulator = simulator

        self.mean = mean
        self.std = std
        self.random_generator = random_generator

        if use_gpu:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = "cpu"
        dtype = torch.double
        self.tensor_kwargs = {"device": device, "dtype": dtype}
        self.iteration = 0


    @abstractmethod
    def check_converge(self):
        pass

    @abstractmethod
    def move_forward(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class MonteCarloSimulation(CornerGenerate):
    def __init__(self,
                 simulator: Simulator,
                 mean: Tensor,
                 std: Tensor,
                 random_generator: SobolEngine,
                 sample_size: int,
                 samples: torch.Tensor = None):
        super().__init__(simulator, mean, std, random_generator)
        self.sample_size = sample_size
        self.sample = samples

    def check_converge(self):
        if self.iteration > 0:
            return True
        return False

    def move_forward(self):
        if self.check_converge():
            return True
        if self.sample is None:
            # draw samples in hyper cube [0,1]^d
            mc_samples = self.random_generator.draw(self.sample_size).to(**self.tensor_kwargs)
            # transform to normal distribution
            self.sample = torch.distributions.Normal(self.mean, self.std).icdf(mc_samples)  # .T).T

        y, std = self.simulator.sim(self.sample)  # GP对simulator进行修改，添加了std输出，不用需要该会
        self.iteration += 1
        return self.sample, y, std

    def reset(self):
        self.iteration = 0

    def load_samples(self, samples):
        self.sample = samples

