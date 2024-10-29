import numpy as np
import torch
from AlpsNetlist import YADO_Alps


class IntegratedSimulator:
    def __init__(self,
                 netlist,
                 torch_para):
        rootPath = "/home/liyunqi/CornerYO/"  # set your work dir, the results will save here
        bash_path = "/home/liyunqi/CornerYO/setup.bash"  # where the setup.bash to be load
        instance = ["pch_mis", "nch_mis", "g45n1svt", "g45p1svt"]
        mismatch = ["toxnmis", "toxpmis",
                    "vthnmis", "vthpmis",
                    "dwnmis", "dwpmis",
                    "dlnmis", "dlpmis"]
        self.torch_para = torch_para
        self.alps_simulator = YADO_Alps(netlist=netlist,
                                        rootPath=rootPath,
                                        bash_path=bash_path,
                                        instance=instance,
                                        mismatch=mismatch)
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
        self.ojb_key = ["dc_gain", "pm", "gbw"]
        self.yield_key = ["dc_gain", "pm", "gbw"]
        # key = ["dc_gain", "pm", "gbw"]

    def list_to_tensor(self, res_list):
        ele_num = len(res_list)
        res = res_list[0]
        for i in range(1, ele_num):
            res = torch.cat((res, res_list[i]), dim=1)
        return res

    def object_simulator(self, x, num_of_sim):
        sim_num = x.shape[0]
        design_para = x * (self.x_ub - self.x_lb) + self.x_lb
        mismatch_para = torch.zeros((sim_num, num_of_sim, 4 * self.instance_num))
        result_list = self.alps_simulator.sim_with_givenParam(design_para=design_para,
                                                              mismatch=mismatch_para,
                                                              resKeys=self.ojb_key)
        res = self.list_to_tensor(result_list).to(**self.torch_para)
        res = -res
        res[:, 1] += 7E-4
        res[:, 1] *= 1E4
        res[:, 0] += 1.4E-8
        res[:, 0] *= 1E9
        res[:, 0] = -res[:, 0]
        return res

    def yield_simulator(self, x, v):
        design_para = x * (self.x_ub - self.x_lb) + self.x_lb
        result_list = self.alps_simulator.sim_with_givenParam(design_para=design_para,
                                                              mismatch=v,
                                                              resKeys=self.yield_key)
        res = self.list_to_tensor(result_list).to(**self.torch_para)
        res[:, 0] -= 60
        res[:, 1] -= 1E6
        return res


if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    device = "cpu"
    tkwargs = {"device": device, "dtype": dtype}
    num_of_x_mc = 1
    num_of_v_sim = 5
    torch.manual_seed(2)
    design_param = torch.rand(num_of_x_mc, 2 * 11) * (1e-6 - 1e-7) + 3e-7  # get x
    mismatch_mc = torch.randn(num_of_x_mc, num_of_v_sim, 4 * 11)

    test = IntegratedSimulator("opamp_spice_new", tkwargs)
    res1 = test.object_simulator(x=design_param, num_of_sim=num_of_v_sim)  # get the nominal process point

    res2 = test.yield_simulator(x=design_param, v=mismatch_mc)
    print(res1)
    print(res2)
