import numpy as np
import torch
from torch import Tensor
from torch.distributions.normal import Normal

# from surrogate_model import *
from abc import ABC, abstractmethod

SCORE_LEARNING_METHODS = ['U', 'EFF', 'LCB', 'SORT']
GP_LEARNING_METHODS = ['U', 'EFF', 'LCB']


# learning function score, larger is better
def u_learning(m: Tensor, s: Tensor):
    return m.abs() / s


def eef_learning(m: Tensor, s: Tensor, fail_val=0):  # generated from GPT
    n, d = m.shape
    epsilon = 2 * s ** 2
    a = fail_val
    normal = Normal(0, 1)
    term1 = (a - m) / s
    term2 = (a - epsilon - m) / s
    term3 = (a + epsilon - m) / s
    Phi_1 = normal.cdf(term1)
    phi_1 = normal.log_prob(term1).exp()
    Phi_2 = normal.cdf(term2)
    phi_2 = normal.log_prob(term2).exp()
    Phi_3 = normal.cdf(term3)
    phi_3 = normal.log_prob(term3).exp()
    result = (m - a) * (2 * Phi_1 - Phi_2 - Phi_3) - s * (2 * phi_1 - phi_2 - phi_3) + Phi_3 - Phi_2
    result = torch.sum(result, dim=1, keepdim=True)
    return result


def efficient_feasible_learning(m: Tensor, s: Tensor):
    return m.abs() / s  # !!!!!!!!!!!!!!!!


def lower_confidence_bound(m: Tensor, s: Tensor, beta):
    return -(m - beta * s)


def convolutional_entropy_infill(m: Tensor, s: Tensor):
    pass  # !!!!!!!!!!!


def sort_learning(m: Tensor):
    return -m


class LearningFunction(ABC):
    def __init__(self, method: str, learning_step: int):
        self.method = method
        self.learning_step = learning_step
        pass

    @abstractmethod
    def learning_candidate(self):
        pass


class ScoreLearningFunction(LearningFunction):
    def __init__(self, method: str, learning_step: int, model_predict, beta=None):
        super().__init__(method, learning_step)
        if self.method not in SCORE_LEARNING_METHODS:
            raise Exception('不支持的学习函数类型')
        self.model_predict = model_predict
        if self.method == 'LCB':
            if beta is None:
                self.beta = 2
            else:
                self.beta = beta

    def learning_candidate(self, return_score = False):
        if self.method in GP_LEARNING_METHODS and isinstance(self.model_predict, tuple):
            if self.method == 'U':
                score = u_learning(self.model_predict[0], self.model_predict[1])
            elif self.method == 'LCB':
                score = lower_confidence_bound(self.model_predict[0], self.model_predict[1], self.beta)
            elif self.method == 'EFF':
                score = efficient_feasible_learning(self.model_predict[0], self.model_predict[1])
        elif self.method == "SORT":
            if isinstance(self.model_predict, tuple):
                score = sort_learning(self.model_predict[0])
            else:
                score = sort_learning(self.model_predict)
        else:
            raise Exception('不支持的学习函数类型-不应进入的分支')

        score = -score
        ind_pick = torch.argsort(score, dim=0)
        if not return_score:
            return ind_pick[0:self.learning_step, 0]
        else:
            return ind_pick[0:self.learning_step, 0], score

    def reset_model_prediction(self, model_prediction):
        self.model_predict = model_prediction


class ConstrainedMinMax(LearningFunction):
    def __init__(self, method: str, learning_step: int):
        super().__init__(method, learning_step)

    def learning_candidate(self):
        pass
