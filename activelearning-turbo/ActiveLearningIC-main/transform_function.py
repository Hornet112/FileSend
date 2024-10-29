import torch
from torch import Tensor
from abc import ABC, abstractmethod


class TransformFunction(ABC):
    def __init__(self, use_gpu=False):
        if use_gpu:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = "cpu"
        dtype = torch.double
        self.tensor_kwargs = {"device": device, "dtype": dtype}

    @abstractmethod
    def x_transform(self, x: Tensor, recalculate=True):
        pass

    @abstractmethod
    def x_restore(self, x_trans: Tensor):
        pass

    @abstractmethod
    def y_transform(self, y: Tensor):
        pass

    @abstractmethod
    def y_restore(self, y: Tensor):
        pass

    @abstractmethod
    def y_std_restore(self, y: Tensor):
        pass

    @abstractmethod
    def data_transform(self, y: Tensor):
        pass


class x_cube_y_standard(TransformFunction):
    def __init__(self, use_gpu=False):
        super().__init__(use_gpu)
        self.x_upper = 0
        self.x_lower = 0
        self.y_m = 0
        self.y_s = 0

    def y_transform(self, y: Tensor):
        self.y_m = torch.mean(y, dim=0)
        self.y_s = torch.std(y, dim=0, unbiased=False)
        self.y_s[self.y_s == 0] = 1
        y_trans = (y - self.y_m) / self.y_s
        return y_trans

    def y_restore(self, y_trans):
        y = y_trans * self.y_s + self.y_m
        return y

    def y_std_restore(self, y_std_trans):
        y_std = self.y_s * y_std_trans
        return y_std

    def x_transform(self, x: Tensor, recalculate=True):
        if recalculate:
            x_upper = self.x_upper = x.max(dim=0).values.reshape(1, -1)
            x_lower = self.x_lower = x.min(dim=0).values.reshape(1, -1)
        else:
            x_upper = self.x_upper
            x_lower = self.x_lower
        x_trans = (x - x_lower) / (x_upper - x_lower)
        return x_trans

    def x_restore(self, x_trans: Tensor):
        x = (self.x_upper - self.x_lower) * x_trans + self.x_lower
        return x

    def data_transform(self, x, y):
        x_trans = self.x_transform(x, recalculate=True)
        y_trans = self.y_transform(y)
        return x_trans, y_trans

