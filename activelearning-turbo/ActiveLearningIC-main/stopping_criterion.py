import numpy as np
import torch
from torch import Tensor
# from surrogate_model import *
from simulator import *
from learning_function import *

'''
Stopping Criterion Section
'''


class ConvergeCriterion(ABC):
    def __init__(self, sim_budget: int):
        self.sim_budget = sim_budget
        self.sim_count = 0

    @abstractmethod
    def check(self):
        pass

    def sim_check(self):
        if self.sim_count < self.sim_budget:
            return False
        else:
            return True

    def update_sim(self, add_sim: int):
        self.sim_count += add_sim


class MaxSimCriterion(ConvergeCriterion):
    def __init__(self, sim_budget: int):
        super().__init__(sim_budget)

    def check(self):
        return self.sim_check()


class CornerCriterion(ConvergeCriterion):
    def __init__(self, sim_budget, delay, corner_len):
        super().__init__(sim_budget)
        self.delay = delay
        self.remain = 0
        self.idx = torch.zeros(0)
        self.previous_idx = torch.zeros(0)
        self.corner_len = corner_len

    def check(self):
        if self.remain == self.delay:
            return True
        if self.sim_count > self.sim_budget + self.corner_len:
            print(f"{self.corner_len*2} more interation did not converge")
            return True
        sort_idx,_ = torch.sort(self.idx, dim=0)
        sort_pre_idx, _ = torch.sort(self.previous_idx, dim=0)
        with open("corner","a") as f :
            f.writelines(f"current: {self.idx.reshape(-1)}\n")
        if torch.equal(sort_idx, sort_pre_idx):
            self.remain += 1
            if not torch.equal(self.idx, self.previous_idx):
                pass
        else:
            self.remain = 0
        return False

    def update_sim(self, add_sim, idx: Tensor = None):
        self.sim_count += add_sim
        self.previous_idx = self.idx
        self.idx = idx


class MaxRank(ConvergeCriterion):
    def __init__(self, sim_budget, test_x, test_y, test_y_predict, i_key, m_th):
        super().__init__(sim_budget)
        self.test_x = test_x
        self.test_y = test_y
        self.test_y_predict = test_y_predict
        self.i_key = i_key
        self.m_th = m_th  # max_rank的阈值

    def check(self):
        if self.sim_count >= self.sim_budget:
            return True
        else:
            y_idx = torch.argsort(self.test_y.squeeze())
            y_p_idx = torch.argsort(self.test_y_predict.squeeze())
            max_rank = 0
            for i in range(self.i_key):
                r = torch.where(y_p_idx == y_idx[i])[0]
                if max_rank <= r:
                    max_rank = r
            print('The maxrank is:', max_rank)
            if max_rank <= self.m_th:
                print('Reached the requirement')
                return True
            else:
                return False


class LearningFunctionValue(ConvergeCriterion):
    def __init__(self, sim_budget, LF_th):
        super().__init__(sim_budget)
        self.LF_th = LF_th

    def check(self, LF_value_set):
        if self.sim_count >= self.sim_budget:
            return True
        else:
            U_min = torch.min(LF_value_set, dim=0)[0]  # It will return (value, idx). We only need the value
            print(U_min)
            if torch.all(U_min > self.LF_th):
                return True
            else:
                return False

    def update_sim(self, add_sim: int):
        self.sim_count += add_sim
