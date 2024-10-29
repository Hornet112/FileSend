import numpy as np
import torch
from AlpsNetlist import YADO_Alps


class IntegratedSimulator:
    def __init__(self,
                 netlist,
                 torch_para):
        self.torch_para = torch_para
        self.alps_simulator = YADO_Alps(netlist)
        self.alps_simulator.load_Design()
        self.instance_num = self.alps_simulator.instance_num

        self.x_lb = torch.ones((self.instance_num * 2)).to(**self.torch_para)
        self.x_lb[0:self.instance_num] = 3.5E-7
        self.x_lb[self.instance_num: 2 * self.instance_num] = 1E-6
        self.x_ub = torch.ones((self.instance_num * 2)).to(**self.torch_para)
        self.x_ub[0:self.instance_num] = 6E-7
        self.x_ub[self.instance_num: 2 * self.instance_num] = 4E-6

        # self.object_simulator = f_obj
        # self.yield_simulator = f_yield_spec
        self.ojb_key = ["delay", "iread"]
        self.yield_key = ["delay", "iread"]
        # key = ["dc_gain", "pm", "gbw"]

    def list_to_tensor(self, res_list):
        ele_num = len(res_list)
        res = res_list[0]
        for i in range(1, ele_num):
            res = torch.cat((res, res_list[i]), dim=1)
        return res

    def object_simulator(self, x):
        sim_num = x.shape[0]
        design_para = x * (self.x_ub - self.x_lb) + self.x_lb
        mismatch_para = torch.zeros((sim_num, 4 * self.instance_num))
        result_list = self.alps_simulator.sim_with_givenParam(design_para, mismatch_para, self.ojb_key)
        res = self.list_to_tensor(result_list).to(**self.torch_para)
        res = -res
        res[:, 1] += 7E-4
        res[:, 1] *= 1E4
        res[:, 0] += 1.4E-8
        res[:, 0] *= 1E9
        res[:, 0] = -res[:, 0]
        return res

    def yield_simulator(self, aug_sample):
        # sim_num = aug_sample.shape[0]
        x = aug_sample[:, 0:2 * self.instance_num]
        design_para = x * (self.x_ub - self.x_lb) + self.x_lb
        mismatch_para = aug_sample[:, 2 * self.instance_num: 6 * self.instance_num]
        result_list = self.alps_simulator.sim_with_givenParam(design_para, mismatch_para, self.yield_key)
        res = self.list_to_tensor(result_list).to(**self.torch_para)
        res[:, 0] -= 60
        res[:, 1] -= 1E6
        return res


if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    device = "cpu"
    tkwargs = {"device": device, "dtype": dtype}

    torch.manual_seed(2)
    random_param = torch.rand(22, 2) * (1e-6 - 1e-7) + 3e-7
    random_param[:, 0] = random_param[:, 1] * torch.rand(1) * 0.2 + 2e-7
    random_param = random_param.T
    random_mismatch = torch.randn(11 * 4, 2)
    random_mismatch = random_mismatch.T

    test = IntegratedSimulator("opamp_spice", tkwargs)
    res1 = test.object_simulator(random_param)

    res2 = test.yield_simulator(torch.cat((random_param, random_mismatch), dim=1))
    print(res1)
    print(res2)
