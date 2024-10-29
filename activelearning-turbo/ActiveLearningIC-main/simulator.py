#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import Tensor
from abc import ABC, abstractmethod
from surrogate_model import SurrogateBaseModel


class Simulator(ABC):
    def __init__(self):
        self.sim_count = 0
        pass

    @abstractmethod
    def sim(self, x):
        pass


class SimulatorFunction(Simulator):
    def __init__(self, function):
        super().__init__()
        self.fun_handle = function

    def sim(self, x: Tensor):
        self.sim_count += x.shape[0]
        return self.fun_handle(x)


class SimulatorModel(Simulator):
    def __init__(self, model: SurrogateBaseModel):
        super().__init__()
        self.model = model

    def sim(self, x):
        self.sim_count += x.shape[0]
        return self.model.predict(x, True)
        # return self.model.predict(x)
